# Projet-MNIST
IA entrainée pour reconnaitre des chiffres sur des images


#Donnees1.py 
Ce fichier permet de telecharger les datasets dont j'aurai besoin ici ceux de MNIST inclu dans Torchvision, il s'agit de 60k de donnees d'entrainement et de 10k de donnes de test


#models.py
Ce fichier contient la classe qui definit la structure de mes donnees


#train.py
Ce fichier permet l'entrainement du modele a la du dataset d'entrainement au terme de celui ci la fonction eval() permet d'evaluer le modele en lui donnant un score a la fin de chaque boucle d'entrainement. A noter que l'import de Donnees1.py aurait evite de re telecharger les datasets dans train.py comme on l'a fait d'ailleurs d'autre fichier on cette erreur.


#test_models.py
Me permet de tester mon modele s'il est bien entraine, ce fichier devient obselete des lors que je peux faire mes tests a l'aide de app.py


#test_installation.py
Permet juste de tester l'intallation de CUDA


#app.py
Ce fichier permet de creer une interface web minimaliste a l'aide de Streamlit qui me permet de test rapidement mon modele.

