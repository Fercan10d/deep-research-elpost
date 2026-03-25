# Deep Research — El Post

Herramienta de investigación profunda para periodistas de El Post (medio de comunicación boliviano).

## Stack

- **Python + Streamlit** (UI web)
- **Perplexity sonar-pro** (búsqueda web vía OpenAI SDK)
- **Claude Sonnet 4** (`claude-sonnet-4-20250514`) (análisis, síntesis, generación de queries)
- **reportlab** (generación de PDF)
- **uv** como package manager (el usuario no tiene Python instalado directamente, usar `uv run`)

## Archivos principales

- `app.py` — UI Streamlit: login, formulario, progreso en vivo, streaming del informe, historial, exportación PDF/MD
- `research_agent.py` — Agente híbrido de dos fases: descubrimiento amplio + profundización caso por caso
- `.streamlit/secrets.toml` — API keys y credenciales de usuarios (NO commitear)
- `.streamlit/config.toml` — Tema visual con colores de marca
- `assets/logo.png` — Logo de El Post
- `requirements.txt` — Dependencias pip
- `data/` — Historial de investigaciones por usuario (JSON)

## Arquitectura del agente

El agente trabaja en dos fases:

1. **Fase 1 (Descubrimiento):** Queries amplias para mapear todos los casos/subtemas. Claude analiza y extrae una lista estructurada de casos individuales. Si la completitud es baja, hace segunda ronda.
2. **Fase 2 (Profundización):** Para cada caso descubierto (máx 10, ordenados por importancia), genera queries ultra-específicas, busca en Perplexity, y Claude extrae narrativa detallada con nombres, fechas, montos y desenlaces. Si la cobertura de un caso es <60%, hace ronda extra.
3. **Verificación cruzada** + resumen ejecutivo + síntesis con streaming.

## Branding El Post

- **Primario:** #3B6BE3
- **Secundario/texto:** #1C3B7A
- **Background:** #F8FAFC
- **Font:** Montserrat (Google Fonts)
- **Iconos:** Lucide SVG inline (NO emojis en ningún lugar)
- **Sidebar:** Gradiente azul oscuro con texto blanco
- Logo invertido (blanco) en sidebar, original en login

## Convenciones

- Todo el UI y los prompts están en **español** (periodistas bolivianos)
- Las queries de búsqueda se generan en español E inglés para mayor cobertura
- No usar emojis en ningún lugar de la interfaz
- Los acentos deben estar correctos en todo el español (investigación, sesión, configuración, etc.)
- El resumen ejecutivo debe ser texto plano sin formato markdown
- El PDF debe renderizar tablas markdown como tablas reales de reportlab

## Ejecución local

```bash
uv run streamlit run app.py
```

La app corre en `localhost:8501`.

## Deploy

Pensado para Streamlit Cloud. Subir repo a GitHub, conectar en share.streamlit.io, pegar secrets.toml en Settings > Secrets.

## Usuarios

5 periodistas con credenciales en `.streamlit/secrets.toml` bajo `[passwords]`.
