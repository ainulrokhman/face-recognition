#!/bin/bash
# Generate self-signed SSL certificate for development/staging
# For production, replace with certificates from a trusted CA (Let's Encrypt, Cloudflare, etc.)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SSL_DIR="$SCRIPT_DIR/ssl"

mkdir -p "$SSL_DIR"

echo "Generating self-signed SSL certificate..."
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "$SSL_DIR/selfsigned.key" \
    -out "$SSL_DIR/selfsigned.crt" \
    -subj "/CN=localhost/O=FaceRecognitionAPI/C=ID"

echo "Certificate generated successfully:"
echo "  Certificate: $SSL_DIR/selfsigned.crt"
echo "  Private Key: $SSL_DIR/selfsigned.key"
echo ""
echo "WARNING: This is a self-signed certificate for development only."
echo "For production, use a certificate from a trusted CA."
