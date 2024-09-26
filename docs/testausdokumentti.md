# Testaus
### NeuralNet luokka:
- Testataan että konstruktori luo oikeanlaisen verkon (kesken).

### Koko verkon toiminta:
- Testataan, että verkon kyky tunnistaa numeroita paranee yhden erän jälkeen.
- Testataan, että virhefunktion arvo pienenee yhden erän jälkeen.
- Testataan että verkko ylisovittaa pienelle määrälle harjoitusdataa (kesken).

### Vastavirta algoritmi:
- Vastavirta algoritmin palauttamat gradientti vektorit muuttuvat eli, eivät sisällä pelkkiä nollia.
- Vastavirta algoritmin laskemat gradientit ovat oikein laskettu. Tämä testataan gradient checkillä, jossa siis lasketaan numeerinen gradientti ja verrataan sitä analyyttiseen gradienttiin.




![image](https://github.com/user-attachments/assets/5cdcc730-9481-4e5a-81a8-8ca11d9acfc3)

