## Data Representativeness

* les projets data science sont dépendants d'un échantillon de test pour évaluer les solutions choisies
* la qualité de l'échantillonnage de ce jeu de test n'est jamais verifié ce qui a des conséquences en terme de reproductibilité, de stabilité des prédictions et de productivité des projets de Data Science 
* notre offre permet de vérifier statistiquement la représentativité du jeu de test par rapport aux données d'origines ce qui permet d'améliorer la qualité d'entrainement et d'évaluation des algorithmes de machine learning
* notre approche permet aussi d'évaluer notre échantillon de test par rapport aux données de production pour identifier les dérives d'étalonnage de la solution 

## Exemples de uses cases

* un projet sur un gros volume de données. 
Une section temporelle a été choisi comme jeu de test. Lors de la mise en prodcution du meilleur model à partir de l'évaluation sur le jeu de test, les resultats sont inférieurs a ceux attendus. Le problème s'explique en partie par le manque de représentativité du jeu de test et du jeu d'entrainement. Suite à la découverte de ce problème, le data scientist doit reprendre toute sa solution pour s'assurer que le meilleur algorithme et les meilleurs paramètres sont toujours valables après un nouvel échantillonnage. Selon le niveau d'industrialisation du projet, ce problème peut indure un surcout de 25% à 100% du temps de dev initial.

* un projet sur une faible quantité de données.
La collecte de nouvelles données est trop couteuse. Le projet limite la quantité de données de tests au minimum. Cette proportion est souvent un choix arbitraire (30 à 10 %). Lors du l'entrainement et l'évaluation de la solution, le data scientist observe une grande variabilité des résultats qu'il attribue aux manque de données. Or si le jeu de test était bien représentatif du jeu d'entrainement on ne devrait pas observer de variation significative de l'évaluation. Cette problématique peut conduire à l'abandon du projet et induire une perte de 100% du temps de dev.

## solution

* données texte: echantillonage après la vectorisation (variables catégorielles)
* images: échantillonage sur les données brutes (variable quantitatives)
* series temporelles: échantillonnage après transfo en fenetres glissantes (variables quantitatives)

les méthodes d'échantillonage sont aussi dépendante de la cible pour les projet de machine learning supervisé.
* classification (variables catégorielles)
* regression (variables quantitatives)

