# Testaus
### NeuralNet luokka:
- Testataan että konstruktori luo oikeanlaisen verkon.

### Koko verkon toiminta:
- Toistaiseksi testaus on toteutettu minisatsi funktiolle, varmistaen että verkon kyky tunnistaa numeroita paranee yhden erän jälkeen.

### Vastavirta algoritmi:
- Vastavirta algoritmin palauttamat gradientti vektorit eivät sisällä pelkkiä nollia.
- Vastavirta algoritmin laskemat gradientit ovat oikein laskettu. Tämä testataan gradient checkillä, jossa siis lasketaan numeerinen gradientti ja verrataan sitä analyyttiseen gradienttiin.


![image](https://github.com/user-attachments/assets/fc8f47a1-13b9-44b9-9a30-4b2254e5c6a8)
