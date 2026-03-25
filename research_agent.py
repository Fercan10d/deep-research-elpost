"""
Agente de Deep Research híbrido: Perplexity (búsqueda) + Claude (análisis y síntesis).
Estrategia de dos fases: descubrimiento amplio + profundización caso por caso.
Optimizado para costos: Haiku para tareas mecánicas, Sonnet solo para análisis crítico.
"""

import json
import time
from dataclasses import dataclass, field
from openai import OpenAI
from anthropic import Anthropic


# ── Modelos ─────────────────────────────────────────────────────────────────
MODEL_HEAVY = "claude-sonnet-4-20250514"   # Análisis complejo y síntesis
MODEL_LIGHT = "claude-haiku-4-5-20251001"  # Generación de queries, tareas mecánicas


@dataclass
class Source:
    url: str
    title: str = ""
    snippet: str = ""


@dataclass
class RoundResult:
    round_number: int
    queries: list[str]
    findings: str
    sources: list[Source]
    gaps: list[str]
    coverage_score: int
    follow_up_queries: list[str]
    phase: str = "discovery"  # "discovery" o "deep_dive"
    case_name: str = ""  # Nombre del caso investigado (solo en deep_dive)


@dataclass
class ResearchConfig:
    """Parámetros del formulario de investigación."""
    topic: str
    angle: str = ""
    scope: str = ""
    source_type: str = "Todos"
    mode: str = "full"  # "full" (dos fases) o "fast" (solo descubrimiento + síntesis)


@dataclass
class ResearchState:
    topic: str
    config: ResearchConfig = None
    rounds: list[RoundResult] = field(default_factory=list)
    final_report: str = ""
    executive_summary: str = ""
    contradictions: list[str] = field(default_factory=list)
    all_sources: list[Source] = field(default_factory=list)
    status: str = "pending"
    discovered_cases: list[dict] = field(default_factory=list)


# ── Prompts ──────────────────────────────────────────────────────────────────

DISCOVERY_QUERY_PROMPT = """Eres un investigador periodístico de élite. Tu objetivo es DESCUBRIR TODOS los casos, eventos, personas e incidentes específicos relacionados con un tema.

Genera consultas de búsqueda diseñadas para MAPEAR el universo completo de casos. No busques información general — busca LISTADOS, CRONOLOGÍAS y REGISTROS de casos específicos.

IMPORTANTE:
- Genera consultas tanto en ESPAÑOL como en INGLÉS (al menos 2 en cada idioma).
- Las consultas deben buscar: listas de casos, cronologías, investigaciones judiciales, nombres de involucrados, incidentes documentados.
- Usa operadores de búsqueda cuando sea útil (comillas para frases exactas, etc.)

{context_instructions}

Responde SOLO con un JSON array de strings. Genera entre 5 y 6 consultas.
Ejemplo: ["lista de casos de corrupción en [tema] 2020-2025", "cronología investigaciones judiciales [tema]", "all corruption cases [topic] timeline", "key figures involved [topic] investigation"]"""

DISCOVERY_ANALYSIS_PROMPT = """Eres un analista de investigación periodística de primer nivel. Tu tarea es analizar los resultados de búsqueda e IDENTIFICAR TODOS LOS CASOS, EVENTOS O SUBTEMAS ESPECÍFICOS que existen dentro del tema investigado.

## Tema: "{topic}"
{angle_instruction}

{context}

Tu trabajo es EXTRAER una lista exhaustiva de casos individuales. Para cada caso necesitas:
1. Un nombre o título identificativo del caso
2. Una descripción breve (1-2 oraciones) de qué se trata
3. Los actores principales (personas, instituciones) involucrados
4. Nivel de importancia (alto/medio/bajo)

Responde SOLO con un JSON con esta estructura exacta:
{{
    "overview": "Panorama general del tema en 2-3 párrafos. Incluye contexto histórico, magnitud del fenómeno y marco general.",
    "discovered_cases": [
        {{
            "name": "Nombre del caso (ej: Caso Fondo Indígena, Caso compra de respiradores)",
            "description": "Descripción breve pero específica — incluye fechas, montos, personas si están disponibles",
            "key_actors": ["Persona 1", "Institución 2"],
            "importance": "alto"
        }}
    ],
    "contradictions": ["contradicción entre fuentes 1"],
    "additional_discovery_queries": ["consulta para descubrir más casos que quizás faltan"],
    "estimated_total_cases": 0,
    "discovery_completeness": 0
}}

INSTRUCCIONES CRÍTICAS:
- NO seas genérico. Cada caso debe tener NOMBRES, FECHAS y DATOS CONCRETOS.
- Si encuentras referencias a casos adicionales que no están bien documentados en los resultados, inclúyelos con lo que tengas.
- "discovery_completeness" es 0-100: qué tan seguro estás de haber identificado TODOS los casos relevantes.
- Prioriza casos por importancia. Los de importancia "alto" serán investigados primero."""

DEEP_DIVE_QUERY_PROMPT = """Genera consultas de búsqueda ultra-específicas para investigar a fondo este caso:

## Caso: {case_name}
## Descripción: {case_description}
## Actores clave: {key_actors}
## Tema general: {topic}

{context_instructions}

Las consultas deben buscar: cronología detallada, nombres completos con cargos, montos exactos, estado de procesos judiciales, declaraciones oficiales, documentos fuente.

Genera en ESPAÑOL e INGLÉS. Responde SOLO con un JSON array de strings. Genera 3 consultas.
Ejemplo: ["nombre persona caso nombre_caso sentencia 2025", "case name court ruling details", "caso nombre cifras auditoria"]"""

CROSS_VERIFICATION_PROMPT = """Eres un verificador de hechos (fact-checker) periodístico de primer nivel. Analiza los siguientes hallazgos recopilados de múltiples fuentes y realiza una verificación cruzada rigurosa.

## Tema: {topic}

## Hallazgos de todas las rondas:
{all_findings}

## Contradicciones detectadas previamente:
{contradictions}

Responde SOLO con un JSON con esta estructura:
{{
    "verified_facts": ["hecho verificado por múltiples fuentes 1", "hecho verificado 2"],
    "disputed_claims": ["afirmación en disputa 1 — Fuente A dice X, Fuente B dice Y"],
    "unverified_claims": ["afirmación que solo aparece en una fuente 1"],
    "reliability_notes": "Evaluación general de la confiabilidad de la información recopilada en 1-2 párrafos"
}}"""

EXECUTIVE_SUMMARY_PROMPT = """Eres un editor periodístico senior. Genera un resumen ejecutivo conciso de la siguiente investigación. Este resumen es para editores que necesitan entender la investigación en 30 segundos.

## Tema: {topic}

## Casos descubiertos: {num_cases}

## Hallazgos principales:
{key_findings}

## Verificación:
{verification}

Escribe un resumen de MÁXIMO 200 palabras en español. Debe responder:
1. ¿Qué se investigó?
2. ¿Cuántos casos/eventos se documentaron?
3. ¿Cuáles son los 3-4 hallazgos más importantes (con datos específicos)?
4. ¿Qué tan confiable es la información?

No uses encabezados, negritas, asteriscos ni ningún formato markdown. Solo texto directo y conciso en prosa corrida. Incluye cifras y nombres cuando sea posible."""

SYNTHESIS_PROMPT = """Eres un redactor de investigación periodística de un medio de comunicación en español. Genera un informe de investigación exhaustivo, detallado y verificable basado en TODOS los hallazgos recopilados.

## Tema de investigación
{topic}

## Contexto del periodista
{research_context}

## Panorama general descubierto
{overview}

## Casos individuales investigados
{cases_detail}

## Verificación cruzada
{verification}

## Fuentes disponibles
{sources_list}

## Instrucciones de estructura

Genera el informe con la siguiente estructura en Markdown:

# {topic}

## Resumen General
(Panorama completo: cuántos casos se documentaron, período analizado, magnitud del fenómeno, actores principales)

## Casos Documentados

(Para CADA caso investigado, genera una subsección con esta estructura:)

### [Número]. [Nombre del Caso]

**Hechos:** Narrativa completa del caso con cronología, nombres, cargos, cifras exactas.

**Actores involucrados:** Lista con nombres completos, cargos y estado actual.

**Cifras clave:** Montos, estadísticas y datos duros.

**Estado actual:** Situación legal/política actual del caso.

(Repite para cada caso)

## Tabla Resumen

(Genera una tabla en Markdown con columnas: Caso | Período | Monto/Magnitud | Involucrados principales | Estado)

## Patrones y Conexiones
(Análisis transversal: patrones comunes entre casos, conexiones entre actores, tendencias temporales)

## Verificación y Confiabilidad
(Datos confirmados por múltiples fuentes, datos en disputa, afirmaciones no verificadas. CRÍTICO para rigor periodístico.)

## Vacíos Informativos y Limitaciones
(Qué no se pudo verificar, qué información falta, qué líneas de investigación quedan abiertas)

## Conclusiones
(Síntesis final: hallazgos más relevantes, implicaciones, recomendaciones para investigación futura)

## Fuentes
(Lista numerada de todas las fuentes utilizadas con URLs)

INSTRUCCIONES CRÍTICAS:
- Cita las fuentes usando [n] donde n es el número de la fuente en la lista final.
- CADA caso debe tener NOMBRES COMPLETOS, FECHAS, MONTOS y DESENLACES.
- No escribas párrafos genéricos. Cada oración debe contener un dato específico y verificable.
- Si un dato no está confirmado, márcalo explícitamente como "no verificado" o "según [fuente única]".
- La tabla resumen es OBLIGATORIA — permite al lector ver el panorama completo de un vistazo.
- Distingue CLARAMENTE entre hechos verificados y afirmaciones no confirmadas.
- Escribe en español formal periodístico.
- No inventes información que no esté en los hallazgos proporcionados.
- Apunta a un informe extenso y detallado. La profundidad es más valiosa que la brevedad."""


# ── Clientes API ─────────────────────────────────────────────────────────────

def create_perplexity_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")


def create_claude_client(api_key: str) -> Anthropic:
    return Anthropic(api_key=api_key)


# ── Helpers ──────────────────────────────────────────────────────────────────

MAX_PERPLEXITY_CHARS = 4000  # Límite por respuesta de Perplexity al almacenar
MAX_CASE_CHARS = 8000       # Límite por caso en síntesis
MAX_VERIFY_CHARS = 60000    # Límite total para verificación cruzada


def _truncate(text: str, max_chars: int) -> str:
    """Trunca texto al límite de caracteres, cortando en el último párrafo completo."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_para = truncated.rfind("\n\n")
    if last_para > max_chars * 0.7:
        truncated = truncated[:last_para]
    return truncated + "\n\n[... texto truncado por extensión]"


def _parse_json_response(text: str) -> dict | list:
    """Extrae JSON de una respuesta de Claude que puede venir envuelta en markdown."""
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def _build_context_instructions(config: ResearchConfig) -> str:
    """Construye instrucciones de contexto para la generación de queries."""
    parts = []
    if config.angle:
        parts.append(f"ÁNGULO ESPECÍFICO: El periodista quiere investigar desde la perspectiva de: {config.angle}. Enfoca las consultas en este ángulo.")
    if config.scope:
        parts.append(f"ALCANCE: Limita la búsqueda a: {config.scope}.")
    if config.source_type and config.source_type != "Todos":
        source_map = {
            "Datos oficiales y gubernamentales": "Prioriza fuentes oficiales, gubernamentales, estadísticas nacionales e internacionales.",
            "Medios de comunicación": "Prioriza cobertura de medios de comunicación, reportajes y noticias.",
            "Informes técnicos y académicos": "Prioriza papers académicos, informes de organizaciones internacionales, estudios técnicos.",
            "Opinión de expertos": "Prioriza entrevistas, columnas de opinión, declaraciones de expertos y analistas.",
        }
        parts.append(source_map.get(config.source_type, ""))
    if not parts:
        parts.append("No hay restricciones adicionales. Busca con amplitud.")
    return "\n".join(parts)


# ── Fase 1: Descubrimiento ──────────────────────────────────────────────────

def generate_discovery_queries(claude: Anthropic, config: ResearchConfig) -> list[str]:
    """Genera queries amplias para descubrir todos los casos/subtemas. [HAIKU]"""
    context_instructions = _build_context_instructions(config)
    system_prompt = DISCOVERY_QUERY_PROMPT.format(context_instructions=context_instructions)

    response = claude.messages.create(
        model=MODEL_LIGHT,
        max_tokens=800,
        temperature=0.4,
        system=system_prompt,
        messages=[{"role": "user", "content": config.topic}],
    )
    return _parse_json_response(response.content[0].text)


def search_perplexity(perplexity: OpenAI, query: str, detailed: bool = False) -> tuple[str, list[Source]]:
    """Busca en Perplexity y devuelve el contenido + fuentes."""
    if detailed:
        system_msg = (
            "Eres un investigador periodístico de élite realizando una investigación profunda. "
            "Tu respuesta debe ser EXTREMADAMENTE detallada y estructurada. "
            "OBLIGATORIO incluir: "
            "1) Nombres completos de TODAS las personas involucradas con sus cargos exactos. "
            "2) Fechas exactas de cada evento (día/mes/año cuando sea posible). "
            "3) Montos y cifras específicas (en dólares y moneda local). "
            "4) Cronología detallada de los hechos. "
            "5) Estado actual: sentencias, detenciones, prófugos, procesos abiertos. "
            "6) Fuentes primarias: documentos judiciales, informes de auditoría, declaraciones oficiales. "
            "No resumas — desarrolla cada punto con la mayor cantidad de datos posible. "
            "Responde en español aunque la consulta sea en inglés."
        )
    else:
        system_msg = (
            "Eres un asistente de investigación periodística de élite. "
            "Responde de forma EXTREMADAMENTE detallada y completa. "
            "SIEMPRE incluye: nombres completos de personas con sus cargos, "
            "fechas exactas, montos y cifras específicas, estado actual de procesos, "
            "fuentes primarias cuando estén disponibles. "
            "Si hay múltiples casos o eventos, LISTA TODOS con detalles individuales. "
            "Responde en español aunque la consulta sea en inglés."
        )

    response = perplexity.chat.completions.create(
        model="sonar-pro",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query},
        ],
    )
    content = response.choices[0].message.content
    sources = []
    if hasattr(response, "citations") and response.citations:
        for url in response.citations:
            sources.append(Source(url=url))
    return content, sources


def analyze_discovery(
    claude: Anthropic, config: ResearchConfig, search_results: list[str]
) -> dict:
    """Claude analiza los resultados de descubrimiento e identifica todos los casos. [SONNET]"""
    context_parts = []
    for i, result in enumerate(search_results, 1):
        context_parts.append(f"### Resultado de búsqueda {i}\n{result}")

    angle_instruction = ""
    if config.angle:
        angle_instruction = f"\nÁngulo de investigación: {config.angle}"

    context = "\n\n".join(context_parts)
    prompt = DISCOVERY_ANALYSIS_PROMPT.format(
        topic=config.topic, context=context, angle_instruction=angle_instruction
    )

    response = claude.messages.create(
        model=MODEL_HEAVY,
        max_tokens=4000,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_json_response(response.content[0].text)


# ── Fase 2: Profundización ──────────────────────────────────────────────────

def generate_deep_dive_queries(
    claude: Anthropic, case: dict, config: ResearchConfig
) -> list[str]:
    """Genera queries específicas para profundizar en un caso individual. [HAIKU]"""
    context_instructions = _build_context_instructions(config)

    prompt = DEEP_DIVE_QUERY_PROMPT.format(
        case_name=case["name"],
        case_description=case.get("description", ""),
        key_actors=", ".join(case.get("key_actors", [])),
        topic=config.topic,
        context_instructions=context_instructions,
    )

    response = claude.messages.create(
        model=MODEL_LIGHT,
        max_tokens=500,
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        queries = _parse_json_response(response.content[0].text)
    except (json.JSONDecodeError, IndexError):
        queries = [f"{case['name']} {config.topic} detalles cronología"]

    return queries


# ── Verificación y síntesis ─────────────────────────────────────────────────

def cross_verify(
    claude: Anthropic, topic: str, all_findings: list[str], contradictions: list[str]
) -> dict:
    """Verificación cruzada de toda la información recopilada. [SONNET]"""
    prompt = CROSS_VERIFICATION_PROMPT.format(
        topic=topic,
        all_findings="\n\n".join(all_findings),
        contradictions="\n".join(f"- {c}" for c in contradictions) if contradictions else "Ninguna detectada aún.",
    )

    response = claude.messages.create(
        model=MODEL_HEAVY,
        max_tokens=3000,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_json_response(response.content[0].text)


def generate_executive_summary(
    claude: Anthropic, topic: str, key_findings: str, verification: dict, num_cases: int
) -> str:
    """Genera el resumen ejecutivo de 30 segundos. [HAIKU]"""
    prompt = EXECUTIVE_SUMMARY_PROMPT.format(
        topic=topic,
        key_findings=key_findings,
        verification=json.dumps(verification, ensure_ascii=False, indent=2),
        num_cases=num_cases,
    )

    response = claude.messages.create(
        model=MODEL_LIGHT,
        max_tokens=400,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def synthesize_report(
    claude: Anthropic, config: ResearchConfig, state: ResearchState, verification: dict
):
    """Claude genera el informe final de investigación con streaming. [SONNET]"""
    all_sources = []
    seen_urls = set()

    for r in state.rounds:
        for s in r.sources:
            if s.url not in seen_urls:
                seen_urls.add(s.url)
                all_sources.append(s)

    sources_list = "\n".join(
        f"[{i+1}] {s.title or s.url} - {s.url}" for i, s in enumerate(all_sources)
    )

    # Construir contexto de investigación
    research_context_parts = []
    if config.angle:
        research_context_parts.append(f"Ángulo: {config.angle}")
    if config.scope:
        research_context_parts.append(f"Alcance: {config.scope}")
    if config.source_type and config.source_type != "Todos":
        research_context_parts.append(f"Tipo de fuentes priorizadas: {config.source_type}")
    research_context = "\n".join(research_context_parts) if research_context_parts else "Sin restricciones específicas."

    # Overview de la fase de descubrimiento
    overview_parts = []
    for r in state.rounds:
        if r.phase == "discovery":
            overview_parts.append(r.findings)
    overview = "\n\n".join(overview_parts) if overview_parts else "No disponible."

    # Detalle de cada caso — usa los hallazgos crudos de Perplexity
    cases_detail_parts = []
    for case in state.discovered_cases:
        case_name = case.get("name", "Sin nombre")
        case_findings = []

        for r in state.rounds:
            if r.phase == "deep_dive" and r.case_name == case_name:
                case_findings.append(r.findings)

        if case_findings:
            combined = "\n\n".join(case_findings)
            combined = _truncate(combined, MAX_CASE_CHARS)
            cases_detail_parts.append(
                f"### {case_name}\n"
                f"**Descripción:** {case.get('description', '')}\n"
                f"**Actores clave:** {', '.join(case.get('key_actors', []))}\n\n"
                f"**Hallazgos de búsqueda:**\n" + combined
            )
        else:
            cases_detail_parts.append(
                f"### {case_name}\n"
                f"**Descripción:** {case.get('description', '')}\n"
                f"**Actores clave:** {', '.join(case.get('key_actors', []))}\n"
                f"*No se realizó profundización individual para este caso.*"
            )

    cases_detail = "\n\n---\n\n".join(cases_detail_parts) if cases_detail_parts else "No se identificaron casos individuales."

    prompt = SYNTHESIS_PROMPT.format(
        topic=config.topic,
        research_context=research_context,
        overview=overview,
        cases_detail=cases_detail,
        verification=json.dumps(verification, ensure_ascii=False, indent=2),
        sources_list=sources_list,
    )

    report_chunks = []
    with claude.messages.stream(
        model=MODEL_HEAVY,
        max_tokens=12000,
        temperature=0.5,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            report_chunks.append(text)
            yield text

    return "".join(report_chunks)


# ── Orquestador principal ───────────────────────────────────────────────────

def run_research(
    config: ResearchConfig,
    anthropic_key: str,
    perplexity_key: str,
    on_status=None,
) -> ResearchState:
    """
    Ejecuta el loop completo de investigación en dos fases.
    Optimizado: Haiku para tareas mecánicas, Perplexity para investigación pesada,
    Sonnet solo para análisis de descubrimiento, verificación y síntesis final.
    """
    claude = create_claude_client(anthropic_key)
    perplexity = create_perplexity_client(perplexity_key)

    state = ResearchState(topic=config.topic, config=config, status="in_progress")

    def status(msg):
        if on_status:
            on_status(msg)

    all_contradictions = []
    round_counter = 0

    # ════════════════════════════════════════════════════════════════════════
    # FASE 1: DESCUBRIMIENTO
    # ════════════════════════════════════════════════════════════════════════
    status("**FASE 1: Descubrimiento**")
    status("Mapeando el universo de casos y subtemas...")

    # Generar queries de descubrimiento [HAIKU]
    round_counter += 1
    status("Generando consultas de descubrimiento (ES + EN)...")
    discovery_queries = generate_discovery_queries(claude, config)
    status(f"Buscando con {len(discovery_queries)} consultas...")

    # Buscar en Perplexity
    discovery_contents = []
    discovery_sources = []
    for query in discovery_queries:
        status(f"&nbsp;&nbsp;&rarr; *{query}*")
        content, sources = search_perplexity(perplexity, query)
        discovery_contents.append(content)
        discovery_sources.extend(sources)
        time.sleep(0.3)

    # Analizar y descubrir casos [SONNET — llamada crítica]
    status("Analizando resultados y descubriendo casos...")
    discovery = analyze_discovery(claude, config, discovery_contents)

    overview = discovery.get("overview", "")
    cases = discovery.get("discovered_cases", [])
    discovery_contradictions = discovery.get("contradictions", [])
    all_contradictions.extend(discovery_contradictions)
    completeness = discovery.get("discovery_completeness", 50)

    state.rounds.append(RoundResult(
        round_number=round_counter,
        queries=discovery_queries,
        findings=overview,
        sources=discovery_sources,
        gaps=discovery.get("additional_discovery_queries", []),
        coverage_score=completeness,
        follow_up_queries=discovery.get("additional_discovery_queries", []),
        phase="discovery",
    ))
    state.all_sources.extend(discovery_sources)

    status(f"Casos descubiertos: **{len(cases)}**")
    for c in cases:
        importance_label = {"alto": "ALTO", "medio": "MEDIO", "bajo": "BAJO"}.get(c.get("importance", ""), "")
        status(f"&nbsp;&nbsp;&bull; {c['name']} [{importance_label}]")

    # Ronda 2 de descubrimiento si la completitud es baja
    additional_queries = discovery.get("additional_discovery_queries", [])
    if completeness < 70 and additional_queries:
        round_counter += 1
        status("Completitud baja — buscando más casos...")

        extra_contents = []
        extra_sources = []
        for query in additional_queries[:4]:
            status(f"&nbsp;&nbsp;&rarr; *{query}*")
            content, sources = search_perplexity(perplexity, query)
            extra_contents.append(content)
            extra_sources.extend(sources)
            time.sleep(0.3)

        # Re-analizar con toda la información [SONNET — solo si completitud baja]
        all_discovery_contents = discovery_contents + extra_contents
        status("Re-analizando con información ampliada...")
        discovery2 = analyze_discovery(claude, config, all_discovery_contents)

        new_cases = discovery2.get("discovered_cases", [])
        existing_names = {c["name"] for c in cases}
        for nc in new_cases:
            if nc["name"] not in existing_names:
                cases.append(nc)
                status(f"&nbsp;&nbsp;&bull; NUEVO: {nc['name']}")

        overview = discovery2.get("overview", overview)
        all_contradictions.extend(discovery2.get("contradictions", []))

        state.rounds.append(RoundResult(
            round_number=round_counter,
            queries=additional_queries[:4],
            findings=discovery2.get("overview", ""),
            sources=extra_sources,
            gaps=[],
            coverage_score=discovery2.get("discovery_completeness", completeness),
            follow_up_queries=[],
            phase="discovery",
        ))
        state.all_sources.extend(extra_sources)

    # Guardar casos descubiertos
    state.discovered_cases = cases
    total_cases = len(cases)
    status(f"**Total de casos identificados: {total_cases}**")

    # ════════════════════════════════════════════════════════════════════════
    # FASE 2: PROFUNDIZACIÓN (solo en modo full)
    # ════════════════════════════════════════════════════════════════════════
    if config.mode == "full":
        status("**FASE 2: Profundización**")
        status("Investigando cada caso en detalle...")

        # Ordenar por importancia
        importance_order = {"alto": 0, "medio": 1, "bajo": 2}
        sorted_cases = sorted(cases, key=lambda c: importance_order.get(c.get("importance", "medio"), 1))

        # Investigar máximo 10 casos
        max_deep_dive = min(len(sorted_cases), 10)
        cases_to_investigate = sorted_cases[:max_deep_dive]

        if len(sorted_cases) > max_deep_dive:
            status(f"Investigando los {max_deep_dive} casos más importantes de {total_cases} descubiertos.")

        for idx, case in enumerate(cases_to_investigate, 1):
            round_counter += 1
            case_name = case["name"]
            status(f"**Caso {idx}/{max_deep_dive}: {case_name}**")

            # Generar queries específicas [HAIKU]
            status("Generando consultas específicas...")
            queries = generate_deep_dive_queries(claude, case, config)

            # Buscar en Perplexity con modo detallado
            case_contents = []
            case_sources = []
            for query in queries:
                status(f"&nbsp;&nbsp;&rarr; *{query}*")
                content, sources = search_perplexity(perplexity, query, detailed=True)
                case_contents.append(content)
                case_sources.extend(sources)
                time.sleep(0.3)

            # Guardar resultados de Perplexity truncados (SIN pasar por Claude)
            truncated_contents = [_truncate(c, MAX_PERPLEXITY_CHARS) for c in case_contents]
            findings_text = "\n\n---\n\n".join(truncated_contents)

            state.rounds.append(RoundResult(
                round_number=round_counter,
                queries=queries,
                findings=findings_text,
                sources=case_sources,
                gaps=[],
                coverage_score=0,
                follow_up_queries=[],
                phase="deep_dive",
                case_name=case_name,
            ))
            state.all_sources.extend(case_sources)

            status(f"Datos recopilados para {case_name}")
    else:
        status("Modo rápido — sintetizando con datos de descubrimiento...")

    # ════════════════════════════════════════════════════════════════════════
    # FASE 3: VERIFICACIÓN Y SÍNTESIS
    # ════════════════════════════════════════════════════════════════════════
    status("**Verificación cruzada de fuentes...**")
    # Limitar texto para no exceder contexto de Claude
    all_findings_text = [r.findings for r in state.rounds]
    verify_text = []
    total_chars = 0
    for f in all_findings_text:
        if total_chars + len(f) > MAX_VERIFY_CHARS:
            verify_text.append(_truncate(f, MAX_VERIFY_CHARS - total_chars))
            break
        verify_text.append(f)
        total_chars += len(f)
    verification = cross_verify(claude, config.topic, verify_text, all_contradictions)
    state.contradictions = all_contradictions

    status("**Generando resumen ejecutivo...**")
    combined_findings = "\n\n".join(all_findings_text)
    state.executive_summary = generate_executive_summary(
        claude, config.topic, combined_findings, verification, len(cases)
    )

    state.status = "synthesizing"
    state.verification = verification
    return state
