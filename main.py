# ADVERTENCIA: Las detecciones pueden ser erroneas o no incluir
# todos los animales presentes.

import os
import cv2
import json
import argparse
from datetime import datetime
from ultralytics import YOLO

# ─── Configuración ────────────────────────────────────────────

CONFIANZA_MINIMA = 0.5

ANIMALES_COLORES = {
    14: ("ave",       (176,  39, 156)),
    15: ("gato",      (255, 150,  33)),
    16: ("perro",     (  0, 109, 255)),
    17: ("caballo",   ( 99,  30, 233)),
    18: ("oveja",     ( 80, 175,  76)),
    19: ("vaca",      ( 37, 168, 249)),
    20: ("elefante",  (212, 188,   0)),
    21: ("oso",       ( 54,  67, 244)),
    22: ("cebra",     ( 74, 195, 139)),
    23: ("jirafa",    ( 34,  87, 255)),
}

ANIMALES_INFO = {
    14: {"nombre": "Ave",      "alimentacion": "Semillas, insectos, frutos",   "habitat": "Bosques, ciudades",   "dato": "Ven luz ultravioleta"},
    15: {"nombre": "Gato",     "alimentacion": "Carnivoro: pollo, pescado",    "habitat": "Domestico",           "dato": "Ronronear reduce estres"},
    16: {"nombre": "Perro",    "alimentacion": "Omnivoro: carne, cereales",    "habitat": "Domestico, campo",    "dato": "Olfato 40x mas potente"},
    17: {"nombre": "Caballo",  "alimentacion": "Herbivoro: pasto, heno",       "habitat": "Praderas, establos",  "dato": "Duermen de pie"},
    18: {"nombre": "Oveja",    "alimentacion": "Herbivora: pasto, hojas",      "habitat": "Praderas, granjas",   "dato": "Reconocen 50 rostros"},
    19: {"nombre": "Vaca",     "alimentacion": "Herbivora: pasto, maiz",       "habitat": "Granjas, pastizales", "dato": "Tienen 4 estómagos"},
    20: {"nombre": "Elefante", "alimentacion": "Herbivoro: ramas, frutas",     "habitat": "Sabana, selva",       "dato": "Come 150 kg al dia"},
    21: {"nombre": "Oso",      "alimentacion": "Omnivoro: peces, bayas, miel", "habitat": "Bosques, montanas",   "dato": "Hibernan 7 meses"},
    22: {"nombre": "Cebra",    "alimentacion": "Herbivora: pasto, cortezas",   "habitat": "Sabana africana",     "dato": "Rayas unicas por individuo"},
    23: {"nombre": "Jirafa",   "alimentacion": "Herbivora: hojas de acacia",   "habitat": "Sabana africana",     "dato": "Animal terrestre mas alto"},
    # ─── Animales extra (también en COCO) ─────────────────────
    77: ("tiburon",  (200,  80,   0)),  # ← estos son objetos COCO
    # ─── Los siguientes requieren modelo custom ───────────────
}

# Animales extra con info
ANIMALES_INFO_EXTRA = {
    # Puedes agregar aquí animales de modelos custom
}

# Unir ambos
ANIMALES_INFO = {**ANIMALES_INFO, **ANIMALES_INFO_EXTRA}


# ─── Funciones ────────────────────────────────────────────────

def obtener_argumentos():
    """
    Permite pasar la imagen y configuracion desde la terminal.
    Uso      : python main.py --imagen foto.jpg --confianza 0.6
    Recibe   : nada
    Devuelve : argumentos parseados
    """
    parser = argparse.ArgumentParser(description="Detector de animales con YOLO")
    parser.add_argument("--imagen",    type=str,   default="prueba.jpg",  help="Ruta de la imagen")
    parser.add_argument("--confianza", type=float, default=0.5,           help="Confianza minima (0-1)")
    parser.add_argument("--modelo",    type=str,   default="yolov8n.pt",  help="Modelo YOLO a usar")
    parser.add_argument("--reporte",   type=str,   default="reporte.json",help="Ruta del reporte JSON")
    return parser.parse_args()


def cargar_modelo(ruta_modelo="yolov8n.pt"):
    """
    Carga el modelo YOLO desde disco.
    Recibe   : ruta al archivo .pt
    Devuelve : modelo listo para usar
    """
    print(f"  Cargando modelo : {ruta_modelo}")
    return YOLO(ruta_modelo)


def cargar_imagen(ruta_imagen):
    """
    Lee una imagen desde disco verificando que exista.
    Recibe   : ruta al archivo de imagen
    Devuelve : imagen en formato OpenCV, o None si no se encuentra
    """
    if not os.path.exists(ruta_imagen):
        print(f"  Error: el archivo '{ruta_imagen}' no existe")
        return None
    return cv2.imread(ruta_imagen)


def detectar_animales(modelo, imagen, confianza_minima=0.5):
    """
    Ejecuta YOLO sobre la imagen y filtra solo animales conocidos.
    Recibe   : modelo YOLO, imagen OpenCV, confianza minima
    Devuelve : lista de dicts con nombre, color, confianza, bbox e info
    """
    results = modelo(imagen, verbose=False)
    detecciones = []

    for box in results[0].boxes:
        clase_id  = int(box.cls)
        confianza = float(box.conf)

        if clase_id not in ANIMALES_COLORES:
            continue
        if confianza < confianza_minima:
            continue

        nombre, color = ANIMALES_COLORES[clase_id]
        info = ANIMALES_INFO.get(clase_id, {
            "nombre":       nombre,
            "alimentacion": "No disponible",
            "habitat":      "No disponible",
            "dato":         "No disponible",
        })
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detecciones.append({
            "nombre":    nombre,
            "confianza": confianza,
            "color":     color,
            "bbox":      (x1, y1, x2, y2),
            "info":      info,
        })

    return detecciones


def dibujar_ficha(imagen, x1, y1, x2, info, color):
    """
    Dibuja una tarjeta informativa compacta sobre el bounding box.
    Recibe   : imagen, coordenadas, info del animal, color
    Devuelve : imagen modificada
    """
    lineas = [
        f"Come: {info['alimentacion'][:22]}",
        f"Hab : {info['habitat'][:22]}",
        f"Dato: {info['dato'][:22]}",
    ]

    fuente     = cv2.FONT_HERSHEY_SIMPLEX
    escala     = 0.35
    grosor     = 1
    padding    = 4
    alto_linea = 14

    ancho_tarjeta = x2 - x1
    alto_tarjeta  = alto_linea * len(lineas) + padding * 2

    tx = x1
    ty = y1 - alto_tarjeta - 30
    if ty < 0:
        ty = y1 + 30

    overlay = imagen.copy()
    cv2.rectangle(overlay, (tx, ty), (tx + ancho_tarjeta, ty + alto_tarjeta), color, -1)
    cv2.addWeighted(overlay, 0.5, imagen, 0.5, 0, imagen)
    cv2.rectangle(imagen,   (tx, ty), (tx + ancho_tarjeta, ty + alto_tarjeta), color, 1)

    for i, linea in enumerate(lineas):
        pos_y = ty + padding + alto_linea * i + 10
        cv2.putText(imagen, linea, (tx + padding, pos_y),
                    fuente, escala, (255, 255, 255), grosor)

    return imagen


def dibujar_detecciones(imagen, detecciones):
    """
    Dibuja bounding boxes, labels y fichas informativas.
    Recibe   : imagen OpenCV, lista de detecciones
    Devuelve : imagen con todo dibujado
    """
    for det in detecciones:
        x1, y1, x2, y2 = det["bbox"]
        color = det["color"]
        label = f"{det['nombre']} {det['confianza']:.0%}"

        cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 1)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(imagen, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(imagen, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        imagen = dibujar_ficha(imagen, x1, y1, x2, det["info"], color)

    # Fecha y hora en esquina inferior izquierda
    fecha = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
    cv2.putText(imagen, fecha, (10, imagen.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return imagen


def imprimir_detecciones(detecciones):
    """
    Muestra en consola un resumen de los animales encontrados.
    Recibe   : lista de detecciones
    Devuelve : nada
    """
    print(f"\n  Animales detectados: {len(detecciones)}")
    print("  " + "─" * 38)
    for det in detecciones:
        info = det["info"]
        print(f"  Animal    : {info['nombre']}")
        print(f"  Confianza : {det['confianza']:.0%}")
        print(f"  Come      : {info['alimentacion']}")
        print(f"  Habitat   : {info['habitat']}")
        print(f"  Curioso   : {info['dato']}")
        print("  " + "─" * 38)


def guardar_reporte(detecciones, ruta="reporte.json"):
    """
    Guarda un reporte JSON con todas las detecciones.
    Recibe   : lista de detecciones, ruta de salida
    Devuelve : nada
    """
    reporte = {
        "fecha": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "total": len(detecciones),
        "detecciones": [
            {
                "animal":         det["nombre"],
                "confianza":      f"{det['confianza']:.0%}",
                "bbox":           det["bbox"],
                "alimentacion":   det["info"]["alimentacion"],
                "habitat":        det["info"]["habitat"],
                "dato_curioso":   det["info"]["dato"],
            }
            for det in detecciones
        ]
    }

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    print(f"  Reporte JSON guardado en: {ruta}")


def guardar_y_mostrar(imagen, ruta_salida="resultado.jpg"):
    """
    Guarda la imagen resultante y la muestra en pantalla.
    Recibe   : imagen con detecciones, ruta donde guardar
    Devuelve : nada
    """
    cv2.imwrite(ruta_salida, imagen)
    print(f"  Imagen guardada en      : {ruta_salida}")

    cv2.namedWindow("Deteccion de animales", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Deteccion de animales", 900, 600)
    cv2.imshow("Deteccion de animales", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─── Programa principal ───────────────────────────────────────

def main():
    args    = obtener_argumentos()
    modelo  = cargar_modelo(args.modelo)
    imagen  = cargar_imagen(args.imagen)

    if imagen is None:
        return

    detecciones = detectar_animales(modelo, imagen, args.confianza)
    imprimir_detecciones(detecciones)
    guardar_reporte(detecciones, args.reporte)

    imagen_resultado = dibujar_detecciones(imagen, detecciones)
    guardar_y_mostrar(imagen_resultado, "resultado.jpg")


if __name__ == "__main__":
    main()

# ─── Uso desde terminal ───────────────────────────────────────
# python main.py
# python main.py --imagen foto.jpg
# python main.py --imagen foto.jpg --confianza 0.7
# python main.py --imagen foto.jpg --modelo yolov8s.pt --reporte mi_reporte.json