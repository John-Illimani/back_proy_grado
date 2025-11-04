import requests
import urllib3
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from urllib.parse import urlparse
from django.core.cache import cache

# 🔇 Desactiva advertencias de certificados no verificados
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@api_view(["GET"])
@permission_classes([AllowAny])
def proxy_convocatorias(request):
    """
    Proxy que recibe una URL (?url=...) y devuelve el HTML del sitio remoto.
    Permite acceder a páginas externas (por ejemplo, convocatorias de universidades)
    saltando restricciones CORS y manejando errores SSL o de conexión.
    """

    # 1️⃣ Obtener y validar la URL
    url = request.GET.get("url")
    if not url:
        return Response({"error": "Falta el parámetro ?url="}, status=400)

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return Response({"error": "Solo se permiten URLs http o https"}, status=400)

    # 2️⃣ Revisar si ya está en caché (por 1 hora)
    cache_key = f"proxy_html:{url}"
    cached_html = cache.get(cache_key)
    if cached_html:
        return Response(
            {"html": cached_html, "cached": True, "origin_url": url, "status_code": 200},
            status=200,
        )

    # 3️⃣ Configurar cabeceras
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    }

    try:
        # 4️⃣ Realizar la solicitud HTTP
        resp = requests.get(
            url,
            headers=headers,
            timeout=20,
            verify=False,  # ⚠️ ignora SSL en desarrollo
            allow_redirects=True,
        )
        status = resp.status_code

        # 5️⃣ Validar respuesta
        if status != 200:
            return Response(
                {"error": f"Respuesta no exitosa ({status}) desde {url}"},
                status=502,
            )

        html = resp.text.strip()
        if not html or "<html" not in html.lower():
            return Response(
                {"error": f"Respuesta vacía o inválida desde {url}", "status_code": status},
                status=502,
            )

        # 6️⃣ Guardar en caché (1 hora)
        cache.set(cache_key, html, timeout=3600)

        return Response(
            {
                "html": html,
                "status_code": status,
                "origin_url": url,
                "cached": False,
            },
            status=200,
        )

    # 7️⃣ Manejo detallado de errores
    except requests.exceptions.SSLError:
        return Response(
            {"error": f"Error SSL al acceder a {url}. El sitio puede tener certificado inválido."},
            status=502,
        )

    except requests.exceptions.ConnectTimeout:
        return Response(
            {"error": f"Tiempo de conexión agotado al acceder a {url}."},
            status=504,
        )

    except requests.exceptions.ConnectionError as e:
        return Response(
            {
                "error": f"No se pudo conectar a {url}. El servidor puede estar caído o bloqueando solicitudes.",
                "detalle": str(e),
            },
            status=502,
        )

    except requests.exceptions.RequestException as e:
        return Response(
            {"error": f"Error HTTP al acceder a {url}: {str(e)}"},
            status=502,
        )

    except Exception as e:
        return Response(
            {"error": f"Error inesperado: {str(e)}"},
            status=500,
        )
