"""
Deep Research — Herramienta de investigación profunda para periodistas de El Post.
Interfaz Streamlit con agente híbrido Perplexity + Claude.
"""

import json
import base64
from datetime import datetime
from pathlib import Path

import streamlit as st
from research_agent import (
    ResearchState,
    ResearchConfig,
    run_research,
    synthesize_report,
    create_claude_client,
)

# ── Configuración de página ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Deep Research - El Post",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SVG Icons (Lucide-style) ─────────────────────────────────────────────────

ICON_SEARCH = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>'
ICON_HISTORY = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/></svg>'
ICON_FILE = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/></svg>'
ICON_LINK = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>'
ICON_SHIELD = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/></svg>'
ICON_ALERT = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>'


# ── Brand CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif;
        color: #1C3B7A;
        letter-spacing: -0.02em;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1C3B7A 0%, #2A4F9E 100%);
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin-bottom: 4px;
        transition: background 0.2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.2);
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2);
    }
    [data-testid="stSidebar"] .stButton > button {
        color: #ffffff !important;
        border-color: rgba(255,255,255,0.3) !important;
        background: rgba(255,255,255,0.1) !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.2) !important;
        border-color: rgba(255,255,255,0.5) !important;
    }
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stButton > button span {
        color: #ffffff !important;
    }

    /* Botones primarios */
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button {
        background: #3B6BE3;
        color: white !important;
        border: none;
        border-radius: 8px;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: background 0.2s;
    }
    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button:hover {
        background: #1C3B7A;
        color: white !important;
    }

    .stButton > button {
        border-radius: 8px;
        font-family: 'Montserrat', sans-serif;
        font-weight: 500;
        border-color: #CBD5E1;
    }

    .stExpander {
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    .stTextArea textarea, .stTextInput input {
        border-radius: 8px;
        border-color: #CBD5E1;
        font-family: 'Montserrat', sans-serif;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #3B6BE3;
        box-shadow: 0 0 0 2px rgba(59,107,227,0.2);
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    [data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-weight: 500;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1C3B7A !important;
        font-weight: 700;
    }

    .stProgress > div > div {
        background-color: #3B6BE3;
    }

    .research-status {
        padding: 0.4rem 0.8rem;
        border-left: 3px solid #3B6BE3;
        margin-bottom: 0.3rem;
        font-size: 0.9rem;
        color: #1C3B7A;
    }

    .stDownloadButton > button {
        background: white;
        border: 2px solid #3B6BE3;
        color: #3B6BE3 !important;
        border-radius: 8px;
        font-weight: 600;
    }
    .stDownloadButton > button:hover {
        background: #3B6BE3;
        color: white !important;
    }

    .source-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.8rem;
        background: white;
        border-radius: 6px;
        border: 1px solid #CBD5E1;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
    }
    .source-item svg { color: #3B6BE3; flex-shrink: 0; }
    .source-item a { color: #3B6BE3; text-decoration: none; }
    .source-item a:hover { text-decoration: underline; }

    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #1C3B7A;
        margin-bottom: 0.5rem;
    }
    .section-header svg { color: #3B6BE3; }

    /* Executive summary box */
    .exec-summary {
        background: linear-gradient(135deg, #EEF2FF 0%, #F8FAFC 100%);
        border: 1px solid #3B6BE3;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
        color: #1C3B7A;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .exec-summary-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #3B6BE3;
        margin-bottom: 0.5rem;
    }

    /* Verification badges */
    .verified-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.2rem 0.6rem;
        background: #ECFDF5;
        border: 1px solid #6EE7B7;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #065F46;
        margin: 0.2rem;
    }
    .disputed-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.2rem 0.6rem;
        background: #FEF3C7;
        border: 1px solid #FCD34D;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #92400E;
        margin: 0.2rem;
    }

    /* Form help text */
    .form-help {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: -0.5rem;
        margin-bottom: 0.8rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_logo_base64() -> str:
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        return base64.b64encode(logo_path.read_bytes()).decode()
    return ""


def generate_pdf(report: str, executive_summary: str, topic: str, sources: list, logo_b64: str) -> bytes:
    """Genera un PDF del informe con el logo de El Post."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, HRFlowable, Table, TableStyle
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
        import io
        import re

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=2.5*cm, rightMargin=2.5*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            'BrandTitle', parent=styles['Title'],
            fontName='Helvetica-Bold', fontSize=18,
            textColor=HexColor('#1C3B7A'), spaceAfter=6,
        ))
        styles.add(ParagraphStyle(
            'BrandH2', parent=styles['Heading2'],
            fontName='Helvetica-Bold', fontSize=14,
            textColor=HexColor('#1C3B7A'), spaceBefore=16, spaceAfter=8,
        ))
        styles.add(ParagraphStyle(
            'BrandH3', parent=styles['Heading3'],
            fontName='Helvetica-Bold', fontSize=11,
            textColor=HexColor('#3B6BE3'), spaceBefore=12, spaceAfter=6,
        ))
        styles.add(ParagraphStyle(
            'BrandBody', parent=styles['Normal'],
            fontName='Helvetica', fontSize=10,
            textColor=HexColor('#1C3B7A'), alignment=TA_JUSTIFY,
            leading=14, spaceAfter=6,
        ))
        styles.add(ParagraphStyle(
            'ExecSummary', parent=styles['Normal'],
            fontName='Helvetica', fontSize=10,
            textColor=HexColor('#1C3B7A'), alignment=TA_JUSTIFY,
            leading=14, backColor=HexColor('#EEF2FF'),
            borderPadding=10, spaceAfter=12,
        ))
        styles.add(ParagraphStyle(
            'BrandMeta', parent=styles['Normal'],
            fontName='Helvetica', fontSize=8,
            textColor=HexColor('#94A3B8'), alignment=TA_CENTER,
        ))

        elements = []

        # Logo
        if logo_b64:
            import base64 as b64module
            logo_data = b64module.b64decode(logo_b64)
            logo_io = io.BytesIO(logo_data)
            elements.append(RLImage(logo_io, width=4*cm, height=1.2*cm))
            elements.append(Spacer(1, 0.3*cm))

        # Metadata
        date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
        elements.append(Paragraph(f"Deep Research — {date_str}", styles['BrandMeta']))
        elements.append(Spacer(1, 0.5*cm))
        elements.append(HRFlowable(width="100%", color=HexColor('#3B6BE3'), thickness=2))
        elements.append(Spacer(1, 0.5*cm))

        # Executive summary
        if executive_summary:
            styles.add(ParagraphStyle(
                'ExecTitle', parent=styles['Normal'],
                fontName='Helvetica-Bold', fontSize=12,
                textColor=HexColor('#1C3B7A'), leading=16,
                spaceBefore=6, spaceAfter=10,
            ))
            elements.append(Paragraph("RESUMEN EJECUTIVO", styles['ExecTitle']))
            safe_summary = executive_summary.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            safe_summary = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', safe_summary)
            safe_summary = re.sub(r'\*(.+?)\*', r'<i>\1</i>', safe_summary)
            elements.append(Paragraph(safe_summary, styles['ExecSummary']))
            elements.append(Spacer(1, 0.3*cm))

        # Convert markdown to paragraphs
        def sanitize(text):
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
            return text

        # Table cell style
        styles.add(ParagraphStyle(
            'TableCell', parent=styles['Normal'],
            fontName='Helvetica', fontSize=7.5,
            textColor=HexColor('#1C3B7A'), alignment=TA_LEFT,
            leading=10, spaceAfter=0,
        ))
        styles.add(ParagraphStyle(
            'TableHeader', parent=styles['Normal'],
            fontName='Helvetica-Bold', fontSize=7.5,
            textColor=HexColor('#FFFFFF'), alignment=TA_LEFT,
            leading=10, spaceAfter=0,
        ))

        def parse_table_row(line):
            """Extrae celdas de una fila de tabla markdown."""
            cells = [c.strip() for c in line.split('|')]
            # Quitar celdas vacías del inicio/fin por los | exteriores
            if cells and cells[0] == '':
                cells = cells[1:]
            if cells and cells[-1] == '':
                cells = cells[:-1]
            return cells

        def is_separator_row(line):
            """Detecta filas separadoras como |---|---|"""
            return bool(re.match(r'^[\s|:-]+$', line.replace('-', '')))

        lines = report.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Detectar tabla markdown (línea con | que no es separador)
            if '|' in line and not line.startswith('#'):
                table_rows = []
                header_row = None
                while i < len(lines) and '|' in lines[i].strip():
                    row_line = lines[i].strip()
                    if is_separator_row(row_line):
                        i += 1
                        continue
                    cells = parse_table_row(row_line)
                    if cells:
                        if header_row is None:
                            header_row = cells
                        else:
                            table_rows.append(cells)
                    i += 1

                if header_row:
                    # Construir tabla con reportlab
                    num_cols = len(header_row)
                    # Header
                    header_paras = [Paragraph(sanitize(c), styles['TableHeader']) for c in header_row]
                    all_rows = [header_paras]
                    # Data rows
                    for row in table_rows:
                        # Ajustar número de columnas
                        while len(row) < num_cols:
                            row.append('')
                        row_paras = [Paragraph(sanitize(c), styles['TableCell']) for c in row[:num_cols]]
                        all_rows.append(row_paras)

                    # Calcular anchos disponibles
                    available_width = A4[0] - 5*cm  # margen izq + der
                    col_width = available_width / num_cols

                    table = Table(all_rows, colWidths=[col_width] * num_cols)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1C3B7A')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CBD5E1')),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#FFFFFF'), HexColor('#F8FAFC')]),
                        ('TOPPADDING', (0, 0), (-1, -1), 4),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                        ('LEFTPADDING', (0, 0), (-1, -1), 6),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ]))
                    elements.append(Spacer(1, 0.3*cm))
                    elements.append(table)
                    elements.append(Spacer(1, 0.3*cm))
                continue

            if not line:
                elements.append(Spacer(1, 0.2*cm))
            elif line.startswith('# '):
                elements.append(Paragraph(sanitize(line[2:]), styles['BrandTitle']))
            elif line.startswith('## '):
                elements.append(Paragraph(sanitize(line[3:]), styles['BrandH2']))
            elif line.startswith('### '):
                elements.append(Paragraph(sanitize(line[4:]), styles['BrandH3']))
            elif line.startswith('- '):
                elements.append(Paragraph(f"&bull; {sanitize(line[2:])}", styles['BrandBody']))
            elif line.startswith('* '):
                elements.append(Paragraph(f"&bull; {sanitize(line[2:])}", styles['BrandBody']))
            else:
                elements.append(Paragraph(sanitize(line), styles['BrandBody']))
            i += 1

        doc.build(elements)
        return buffer.getvalue()
    except ImportError:
        return None


# ── Datos persistentes ───────────────────────────────────────────────────────

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def save_research(username: str, state: ResearchState):
    user_dir = DATA_DIR / username
    user_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c if c.isalnum() or c in " -_" else "" for c in state.topic)[:50]
    filename = f"{timestamp}_{safe_topic}.json"
    data = {
        "topic": state.topic,
        "config": {
            "angle": state.config.angle if state.config else "",
            "scope": state.config.scope if state.config else "",
            "source_type": state.config.source_type if state.config else "",
        },
        "timestamp": timestamp,
        "date": datetime.now().isoformat(),
        "rounds": len(state.rounds),
        "final_report": state.final_report,
        "executive_summary": state.executive_summary,
        "contradictions": state.contradictions,
        "verification": getattr(state, "verification", {}),
        "sources": [
            {"url": s.url, "title": s.title, "snippet": s.snippet}
            for s in state.all_sources
        ],
        "round_details": [
            {
                "round": r.round_number,
                "queries": r.queries,
                "findings": r.findings,
                "coverage_score": r.coverage_score,
                "gaps": r.gaps,
            }
            for r in state.rounds
        ],
    }
    (user_dir / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_research_history(username: str) -> list[dict]:
    user_dir = DATA_DIR / username
    if not user_dir.exists():
        return []
    history = []
    for f in sorted(user_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["filename"] = f.name
            history.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return history


# ── Autenticación ────────────────────────────────────────────────────────────

def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = ""

    if st.session_state.authenticated:
        return True

    logo_b64 = get_logo_base64()

    st.markdown(f"""
    <div style="text-align: center; padding: 3rem 0 1rem 0;">
        <img src="data:image/png;base64,{logo_b64}" style="height: 60px; margin-bottom: 1rem;" alt="El Post">
        <h2 style="color: #1C3B7A; font-weight: 700; margin: 0;">Deep Research</h2>
        <p style="color: #94A3B8; font-size: 1rem; margin-top: 0.5rem;">Investigación profunda para periodistas</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2, 1, 1.2])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Usuario", placeholder="Tu nombre de usuario")
            password = st.text_input("Contraseña", type="password")
            submitted = st.form_submit_button("Iniciar Sesión", use_container_width=True)

            if submitted:
                passwords = st.secrets.get("passwords", {})
                if username in passwords and passwords[username] == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos.")
    return False


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        logo_b64 = get_logo_base64()
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0;">
            <img src="data:image/png;base64,{logo_b64}" style="height: 40px; filter: brightness(0) invert(1); margin-bottom: 0.5rem;" alt="El Post">
            <div style="font-size: 0.85rem; font-weight: 600; opacity: 0.8; letter-spacing: 0.05em;">DEEP RESEARCH</div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"Sesión: **{st.session_state.username}**")
        st.divider()

        page = st.radio(
            "Navegación",
            ["Nueva Investigación", "Historial"],
            label_visibility="collapsed",
        )

        st.divider()
        if st.button("Cerrar Sesión", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()

    return page


# ── Página: Nueva Investigación ──────────────────────────────────────────────

def render_new_research():
    st.markdown(f"""
    <h1 style="margin-bottom: 0.2rem;">
        <span style="display: inline-flex; align-items: center; gap: 0.5rem;">
            {ICON_SEARCH} Nueva Investigación
        </span>
    </h1>
    <p style="color: #94A3B8; margin-bottom: 1.5rem;">Completa el formulario para iniciar una investigación. Solo el tema es obligatorio.</p>
    """, unsafe_allow_html=True)

    # Formulario de investigación
    topic = st.text_area(
        "Tema de investigación *",
        placeholder="Ej: ¿Cuál es el estado actual de la inteligencia artificial en América Latina?",
        height=80,
    )
    st.markdown('<div class="form-help">Describe el tema, pregunta o hipótesis que deseas investigar. Sé lo más específico posible para obtener mejores resultados.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        angle = st.text_input(
            "Ángulo o enfoque",
            placeholder="Ej: Impacto ambiental, aspecto económico, derechos humanos...",
        )
        st.markdown('<div class="form-help">Opcional. Indica desde qué perspectiva quieres abordar el tema. Esto enfoca la búsqueda en un ángulo específico en lugar de investigar el tema de forma general.</div>', unsafe_allow_html=True)

    with col2:
        scope = st.text_input(
            "Alcance geográfico o temporal",
            placeholder="Ej: Bolivia 2024-2025, América Latina, últimos 6 meses...",
        )
        st.markdown('<div class="form-help">Opcional. Limita la investigación a un país, región o período de tiempo. Sin esto, el agente buscará información global sin restricción temporal.</div>', unsafe_allow_html=True)

    source_type = st.selectbox(
        "Tipo de fuentes a priorizar",
        ["Todos", "Datos oficiales y gubernamentales", "Medios de comunicación", "Informes técnicos y académicos", "Opinión de expertos"],
        index=0,
    )
    st.markdown('<div class="form-help">Opcional. Selecciona qué tipo de fuentes prefieres. "Todos" busca sin restricción.</div>', unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns([1, 4])
    with col1:
        start = st.button("Investigar", type="primary", use_container_width=True, disabled=not topic)

    if start and topic:
        config = ResearchConfig(
            topic=topic.strip(),
            angle=angle.strip(),
            scope=scope.strip(),
            source_type=source_type,
            mode="fast",
        )
        run_investigation(config)


def run_investigation(config: ResearchConfig):
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    perplexity_key = st.secrets.get("PERPLEXITY_API_KEY", "")

    if not anthropic_key or not perplexity_key:
        st.error("Faltan las API keys. Configura ANTHROPIC_API_KEY y PERPLEXITY_API_KEY en los secrets.")
        return

    status_container = st.container()
    progress_bar = st.progress(0, text="Iniciando investigación...")

    def on_status(msg: str):
        with status_container:
            st.markdown(f'<div class="research-status">{msg}</div>', unsafe_allow_html=True)

    with st.spinner("Investigando..."):
        state = run_research(
            config=config,
            anthropic_key=anthropic_key,
            perplexity_key=perplexity_key,
            on_status=on_status,
        )

    total_rounds = len(state.rounds)
    progress_bar.progress(0.7, text="Generando informe final...")

    # Resumen ejecutivo
    st.markdown("---")
    if state.executive_summary:
        import re as re_mod
        summary_html = state.executive_summary.replace('**', '')  # Limpiar markdown bold
        st.markdown(f"""
        <div class="exec-summary">
            <div class="exec-summary-label">Resumen ejecutivo — 30 segundos</div>
            {summary_html}
        </div>
        """, unsafe_allow_html=True)

    # Verificación cruzada visual
    verification = getattr(state, "verification", {})
    if verification:
        verified = verification.get("verified_facts", [])
        disputed = verification.get("disputed_claims", [])
        if verified or disputed:
            st.markdown(f'<div class="section-header"><h3>{ICON_SHIELD} Verificación cruzada</h3></div>', unsafe_allow_html=True)
            if verified:
                for fact in verified[:5]:
                    st.markdown(f'<span class="verified-badge">{ICON_SHIELD} {fact}</span>', unsafe_allow_html=True)
            if disputed:
                for claim in disputed[:5]:
                    st.markdown(f'<span class="disputed-badge">{ICON_ALERT} {claim}</span>', unsafe_allow_html=True)
            st.markdown("")

    # Informe con streaming
    st.markdown(f'<div class="section-header"><h2>{ICON_FILE} Informe de Investigación</h2></div>', unsafe_allow_html=True)

    report_placeholder = st.empty()
    report_parts = []

    claude = create_claude_client(anthropic_key)
    for chunk in synthesize_report(claude, config, state, verification):
        report_parts.append(chunk)
        report_placeholder.markdown("".join(report_parts))

    state.final_report = "".join(report_parts)
    state.status = "completed"
    progress_bar.progress(1.0, text="Investigación completada.")

    # Guardar
    save_research(st.session_state.username, state)

    # Fuentes
    st.markdown("---")
    st.markdown(f'<div class="section-header"><h3>{ICON_LINK} Fuentes utilizadas</h3></div>', unsafe_allow_html=True)
    seen = set()
    for s in state.all_sources:
        if s.url not in seen:
            seen.add(s.url)
            domain = s.url.split("//")[-1].split("/")[0] if "//" in s.url else s.url
            st.markdown(
                f'<div class="source-item">{ICON_LINK} <a href="{s.url}" target="_blank">{domain}</a></div>',
                unsafe_allow_html=True,
            )

    # Métricas
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rondas", total_rounds)
    col2.metric("Fuentes", len(seen))
    col3.metric(
        "Cobertura",
        f"{state.rounds[-1].coverage_score}%" if state.rounds else "N/A",
    )
    col4.metric(
        "Contradicciones",
        len(state.contradictions),
    )

    # Descargas
    st.markdown("")
    dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 3])
    with dl_col1:
        st.download_button(
            "Descargar .md",
            data=state.final_report,
            file_name=f"investigacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )
    with dl_col2:
        logo_b64 = get_logo_base64()
        pdf_data = generate_pdf(
            state.final_report,
            state.executive_summary,
            config.topic,
            [s for s in state.all_sources if s.url in seen],
            logo_b64,
        )
        if pdf_data:
            st.download_button(
                "Descargar .pdf",
                data=pdf_data,
                file_name=f"investigacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
            )


# ── Página: Historial ────────────────────────────────────────────────────────

def render_history():
    st.markdown(f"""
    <h1 style="margin-bottom: 0.2rem;">
        <span style="display: inline-flex; align-items: center; gap: 0.5rem;">
            {ICON_HISTORY} Historial de Investigaciones
        </span>
    </h1>
    <p style="color: #94A3B8; margin-bottom: 1.5rem;">Tus investigaciones anteriores y sus informes.</p>
    """, unsafe_allow_html=True)

    history = load_research_history(st.session_state.username)

    if not history:
        st.info("No tienes investigaciones previas. Inicia una nueva investigación desde el menú lateral.")
        return

    for item in history:
        date_str = item.get("date", "")
        try:
            date_display = datetime.fromisoformat(date_str).strftime("%d/%m/%Y %H:%M")
        except (ValueError, TypeError):
            date_display = date_str

        num_sources = len(item.get("sources", []))
        coverage = (
            f"{item['round_details'][-1]['coverage_score']}%"
            if item.get("round_details")
            else "N/A"
        )

        with st.expander(f"{item['topic'][:80]}  /  {date_display}  /  {num_sources} fuentes  /  {coverage} cobertura"):
            # Resumen ejecutivo
            if item.get("executive_summary"):
                hist_summary = item['executive_summary'].replace('**', '')
                st.markdown(f"""
                <div class="exec-summary">
                    <div class="exec-summary-label">Resumen ejecutivo</div>
                    {hist_summary}
                </div>
                """, unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rondas", item.get("rounds", "?"))
            col2.metric("Fuentes", num_sources)
            col3.metric("Cobertura", coverage)
            col4.metric("Contradicciones", len(item.get("contradictions", [])))

            if item.get("final_report"):
                st.markdown("---")
                st.markdown(item["final_report"])

                dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 3])
                with dl_col1:
                    st.download_button(
                        "Descargar .md",
                        data=item["final_report"],
                        file_name=f"investigacion_{item.get('timestamp', 'report')}.md",
                        mime="text/markdown",
                        key=f"dl_md_{item.get('filename', '')}",
                    )
                with dl_col2:
                    logo_b64 = get_logo_base64()
                    pdf_data = generate_pdf(
                        item["final_report"],
                        item.get("executive_summary", ""),
                        item["topic"],
                        item.get("sources", []),
                        logo_b64,
                    )
                    if pdf_data:
                        st.download_button(
                            "Descargar .pdf",
                            data=pdf_data,
                            file_name=f"investigacion_{item.get('timestamp', 'report')}.pdf",
                            mime="application/pdf",
                            key=f"dl_pdf_{item.get('filename', '')}",
                        )

            if item.get("sources"):
                st.markdown("---")
                st.markdown("**Fuentes:**")
                for s in item["sources"]:
                    url = s.get("url", "")
                    if url:
                        domain = url.split("//")[-1].split("/")[0] if "//" in url else url
                        st.markdown(
                            f'<div class="source-item">{ICON_LINK} <a href="{url}" target="_blank">{domain}</a></div>',
                            unsafe_allow_html=True,
                        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not check_login():
        return

    page = render_sidebar()

    if page == "Nueva Investigación":
        render_new_research()
    elif page == "Historial":
        render_history()


if __name__ == "__main__":
    main()
