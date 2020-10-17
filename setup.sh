mkdir -p ~/.streamlit/
echo "[general]
email = \"grantdpittman@gmail.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = 8501
enableCORS = false
" > ~/.streamlit/config.toml
