# Määrittelydokumentti
Tämä määrittelydokumentti määrittelee Helsingin yliopiston Aineopintojen harjoitustyö: Algoritmit ja tekoäly -kurssilla tehtävän harjoitustyön. Suoritan kurssin Tietojenkäsittelytieteen kandiohjelmassa (TKT).
## Aihe
Koneoppiminen, hahmontunnistus, käsin kirjoitettujen numeroiden tunnistus käyttämällä neuroverkkoa.
Toteutan neuroverkon opettamisen käyttämällä mini-satsi gradientti menetelmää. Ohjelma ensin oppii harjoitusdatalla ja sitten osaa tunnistaa käsin kirjoitettuja numeroita.

### Syötteet
Mnist tietokanta.
Tietokanta koostuu harmaasävykuvista 28x28 pikseliä, jotka sitten on tarkotus muuttaa 784 ulotteiseksi vektoriksi ja syöttää ohjelmalle.

### Aikavaativuudet
#### Neuroverkot:

- Vastavirta algoritmin aikavaativuus neuroverkkojen kouluttamisessa on O(n^2 * p), missä n on neuroverkon yhteyksien (painojen) lukumäärä ja p on koulutusesimerkkien lukumäärä.
- Stokastinen gradientti menetelmä O(n*((s/k)*m)), missä n on harjotusdatan läpikäyntien määrä, s/k = harjoitusdatan koko/satsin koko ja m on minisatsin aikavaativuus.
- Mini-satsi O(n* m* bp) missä n on minisatsin koko, m on painojen ja vakiotermien määrä ja bp on vastavirta algon aikavaativuus.


## Ohjelmointikielet
Projekti olisi tarkoitus koodata pythonilla.
Vertaisarvoinnit onnistuu myös c++

## Algoritmit
Vastavirta-algoritmi.
Mini-satsi gradientti menetelmä

## Dokumentaatio
Dokumentaatio on suomeksi mutta koodaaminen ja docstringit englanniksi
## Lähteet
https://rafayqayyum.medium.com/computational-complexity-of-machine-learning-algorithms-254c275de84
http://neuralnetworksanddeeplearning.com/chap1.html#a_simple_network_to_classify_handwritten_digits


