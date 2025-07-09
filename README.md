<<<<<<< HEAD
Description du projet :

Ce projet porte sur la détection automatique d’objets à partir d’une image satellite de haute résolution (40×40 km) a 4 bandes. Les trois premières correspondent aux composantes RGB et la quatrième bande contient la matrice des coordonnées spatiales réelles. Le pipeline développé réalise les étapes suivantes :

Découpage de l’image :
L’image satellite est découpée en tuiles de 640×640 pixels avec un chevauchement de 25% pour éviter la perte d’information en bordure des objets.

Annotation automatique avec Florence-2 :
Chaque tuile est analysée par un LLM visuel (Florence-2) qui génère des annotations textuelles. Ces annotations sont ensuite converties au format YOLO pour une utilisation dans des modèles plus rapides.

Fine-tuning de YOLOv8s :
Un modèle YOLOv8s est entraîné sur les annotations générées pour obtenir une détection plus fluide, rapide et légère que Florence-2, tout en conservant de bonnes performances.

Génération automatique d’un rapport :
Un script génère un rapport .txt listant tous les objets détectés dans l’image, avec :

Leur nom

Leur position (en pixels) dans l’image globale

NB: Dans une prochaine version, ces positions seront converties en coordonnées géographiques exactes à l’aide des métadonnées de l’image satellite.
=======
# sat_detection
>>>>>>> b07f7aaa (Initial commit)
