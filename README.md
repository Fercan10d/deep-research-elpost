# 🔬 Deep Research

Herramienta de investigación profunda para periodistas. Usa Perplexity para búsqueda web y Claude para análisis y síntesis.

## Cómo funciona

1. El periodista ingresa un tema de investigación
2. Claude genera consultas de búsqueda estratégicas
3. Perplexity busca en la web y recopila fuentes
4. Claude analiza los hallazgos e identifica vacíos informativos
5. Se repite el ciclo hasta alcanzar cobertura suficiente (máx. 4 rondas)
6. Claude genera un informe estructurado con citas verificables

## Setup local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar secrets
# Editar .streamlit/secrets.toml con tus API keys y usuarios

# Ejecutar
streamlit run app.py
```

## Configurar usuarios

Editar `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
PERPLEXITY_API_KEY = "pplx-..."

[passwords]
juan = "clave_segura_1"
maria = "clave_segura_2"
```

## Deploy en Streamlit Cloud

1. Sube el repo a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta el repo
4. En Settings → Secrets, pega el contenido de `secrets.toml`
5. Listo
