# Määrittelydokumentti
Tämä määrittelydokumentti määrittelee Helsingin yliopiston Aineopintojen harjoitustyö: Algoritmit ja tekoäly -kurssilla tehtävän harjoitustyön. Suoritan kurssin Tietojenkäsittelytieteen kandiohjelmassa (TKT).
## Aihe
Koneoppiminen, hahmontunnistus, käsin kirjoitettujen numeroiden tunnistus käyttämällä neuroverkkoja tai k-lähimmän naapurin menetelmää.

### Syötteet
Mnist tietokanta

### Aikavaativuudet
#### Neuroverkot:

Ymmärtääkseni ainakin backpropagation algoritmin aikavaativuus neuroverkkojen kouluttamisessa on O(n^2 * p), missä n on neuroverkon yhteyksien (painojen) lukumäärä ja p on koulutusesimerkkien lukumäärä.
Nopealla vilkasulla käytetään myös stochastic gradient descent algoritmia mutta aikavaativuuksia en algoritmille muuten löytänyt kun että nopealla omalla analyysillä ainakin O(n)

#### K-lähimmät naapurit:

n = Koulutusesimerkkien lukumäärä 
f = Ominaisuuksien lukumäärä 
k = k lähintä naapuria

Brute force (raakavoima): 
mallin opettamisen aikavaativuus: O(1)
Ennustamisen aikavaativuus: O(nf + kf)
Suoritusaikaisen tilan vaativuus: O(n*f)

kd-puu (kd-tree): 
mallin opettamisen aikavaativuus: O(fnlog(n))
Ennustamisen aikavaativuus: O(klog(n)) 
Suoritusaikaisen tilan vaativuus: O(nf)

## Ohjelmointikielet
Projekti olisi tarkoitus koodata pythonilla.
Vertaisarvoinnit onnistuu myös c++

## Algoritmit
neuroverkko vastavirta-algoritmeineen ym. toteutetaan itse.
tai k-lähimmän naapurin menetelmä.

## Dokumentaatio
Dokumentaatio on suomeksi mutta koodaaminen ja docstringit englanniksi
## Lähteet
https://rafayqayyum.medium.com/computational-complexity-of-machine-learning-algorithms-254c275de84
http://neuralnetworksanddeeplearning.com/chap1.html#a_simple_network_to_classify_handwritten_digits


