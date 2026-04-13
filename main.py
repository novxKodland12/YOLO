#ADVERTENCIA: Las detecciones pueden ser erroneas o no incluir todos los animales presentes. Mas en adelante se muestra como mejorar el modelo para casos específicos.

import cv2
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
}


# ─── Funciones ────────────────────────────────────────────────

def cargar_modelo(ruta_modelo="yolov8n.pt"):
    """
    Carga el modelo YOLO desde disco.
    Recibe : ruta al archivo .pt
    Devuelve: modelo listo para usar
    """
    print(f"Cargando modelo: {ruta_modelo}")
    return YOLO(ruta_modelo)


def cargar_imagen(ruta_imagen):
    """
    Lee una imagen desde disco.
    Recibe : ruta al archivo de imagen
    Devuelve: imagen en formato OpenCV, o None si no se encuentra
    """
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: no se encontro la imagen en '{ruta_imagen}'")
    return imagen


def detectar_animales(modelo, imagen):
    """
    Ejecuta YOLO sobre la imagen y filtra solo animales conocidos.
    Recibe : modelo YOLO, imagen OpenCV
    Devuelve: lista de dicts con nombre, color, confianza, bbox e info
    """
    results = modelo(imagen, verbose=False)
    detecciones = []

    for box in results[0].boxes:
        clase_id  = int(box.cls)
        confianza = float(box.conf)

        if clase_id not in ANIMALES_COLORES:
            continue
        if confianza < CONFIANZA_MINIMA:
            continue

        nombre, color = ANIMALES_COLORES[clase_id]
        info = ANIMALES_INFO[clase_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detecciones.append({
            "nombre":    nombre,
            "confianza": confianza,
            "color":     color,
            "bbox":      (x1, y1, x2, y2),
            "info":      info,
        })

    return detecciones


def dibujar_ficha(imagen, x1, y1, info, color):
    """
    Dibuja una tarjeta informativa a la derecha del bounding box.
    Recibe : imagen, coordenadas del box, info del animal, color
    Devuelve: imagen modificada
    """
    alto_imagen = imagen.shape[0]

    lineas = [
        f"Come  : {info['alimentacion']}",
        f"Hab.  : {info['habitat']}",
        f"Dato  : {info['dato']}",
    ]

    fuente     = cv2.FONT_HERSHEY_SIMPLEX
    escala     = 0.45
    grosor     = 1
    padding    = 6
    alto_linea = 18

    # Ancho de la tarjeta según la línea más larga
    ancho_tarjeta = max(
        cv2.getTextSize(l, fuente, escala, grosor)[0][0]
        for l in lineas
    ) + padding * 2

    alto_tarjeta = alto_linea * len(lineas) + padding * 2

    # Posición: a la derecha del box, sin salirse de la imagen
    tx = x1
    ty = max(0, y1 - alto_tarjeta - 36)  # encima del label principal

    # Fondo semitransparente de la tarjeta
    overlay = imagen.copy()
    cv2.rectangle(overlay,
                  (tx, ty),
                  (tx + ancho_tarjeta, ty + alto_tarjeta),
                  color, -1)
    cv2.addWeighted(overlay, 0.4, imagen, 0.6, 0, imagen)

    # Borde de la tarjeta
    cv2.rectangle(imagen,
                  (tx, ty),
                  (tx + ancho_tarjeta, ty + alto_tarjeta),
                  color, 1)

    # Líneas de texto
    for i, linea in enumerate(lineas):
        pos_y = ty + padding + alto_linea * i + 12
        cv2.putText(imagen, linea, (tx + padding, pos_y),
                    fuente, escala, (255, 255, 255), grosor)

    return imagen

def dibujar_ficha(imagen, x1, y1, x2, info, color):
    """
    Dibuja una tarjeta informativa compacta debajo del label.
    """
    lineas = [
        f"Come: {info['alimentacion'][:20]}",  # ← recorta texto largo
        f"Hab: {info['habitat'][:20]}",
        f"Dato: {info['dato'][:20]}",
    ]

    fuente     = cv2.FONT_HERSHEY_SIMPLEX
    escala     = 0.35   # ← más pequeño (antes 0.45)
    grosor     = 1
    padding    = 4      # ← menos padding (antes 6)
    alto_linea = 14     # ← menos alto (antes 18)

    # Ancho fijo igual al bounding box del animal
    ancho_tarjeta = x2 - x1

    alto_tarjeta = alto_linea * len(lineas) + padding * 2

    # Posición: justo debajo del bounding box
    tx = x1
    ty = y1 - alto_tarjeta - 30  # encima del label

    if ty < 0:
        ty = y1 + 30  # si no cabe arriba, va abajo

    # Fondo semitransparente
    overlay = imagen.copy()
    cv2.rectangle(overlay,
                  (tx, ty),
                  (tx + ancho_tarjeta, ty + alto_tarjeta),
                  color, -1)
    cv2.addWeighted(overlay, 0.5, imagen, 0.5, 0, imagen)

    # Borde
    cv2.rectangle(imagen,
                  (tx, ty),
                  (tx + ancho_tarjeta, ty + alto_tarjeta),
                  color, 1)

    # Texto
    for i, linea in enumerate(lineas):
        pos_y = ty + padding + alto_linea * i + 10
        cv2.putText(imagen, linea, (tx + padding, pos_y),
                    fuente, escala, (255, 255, 255), grosor)

    return imagen


def dibujar_detecciones(imagen, detecciones):
    for det in detecciones:
        x1, y1, x2, y2 = det["bbox"]
        color  = det["color"]
        label  = f"{det['nombre']} {det['confianza']:.0%}"

        # Caja del animal
        cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 1)

        # Label de nombre
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(imagen, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(imagen, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Tarjeta informativa compacta
        imagen = dibujar_ficha(imagen, x1, y1, x2, det["info"], color)

    return imagen


def imprimir_detecciones(detecciones):
    """
    Muestra en consola un resumen de los animales encontrados.
    Recibe : lista de detecciones
    Devuelve: nada
    """
    print(f"\nAnimales detectados: {len(detecciones)}")
    print("─" * 40)
    for det in detecciones:
        info = det["info"]
        print(f"  Animal     : {info['nombre']}")
        print(f"  Confianza  : {det['confianza']:.0%}")
        print(f"  Come       : {info['alimentacion']}")
        print(f"  Habitat    : {info['habitat']}")
        print(f"  Curioso    : {info['dato']}")
        print("─" * 40)




def guardar_y_mostrar(imagen, ruta_salida="resultado.jpg"):
    """
    Guarda la imagen resultante y la muestra en pantalla.
    Recibe : imagen con detecciones, ruta donde guardar
    Devuelve: nada
    """
    cv2.imwrite(ruta_salida, imagen)
    print(f"\nResultado guardado en: {ruta_salida}")
    cv2.imshow("Deteccion de animales", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─── Programa principal ───────────────────────────────────────

def main():
    modelo  = cargar_modelo("yolov8n.pt")
    imagen  = cargar_imagen("prueba.jpg")

    if imagen is None:
        return

    detecciones = detectar_animales(modelo, imagen)
    imprimir_detecciones(detecciones)

    imagen_resultado = dibujar_detecciones(imagen, detecciones)
    guardar_y_mostrar(imagen_resultado, "resultado.jpg")


if __name__ == "__main__":
    main()


#usar "python main.py" para ejecutar el programa. Asegúrate de tener "yolov8n.pt" y "prueba.jpg" en el mismo directorio.