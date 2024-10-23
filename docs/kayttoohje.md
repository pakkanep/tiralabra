Miten ohjelma suoritetaan, miten eri toiminnallisuuksia käytetään.
## Asennus ja käyttö
Asenna riippuvuudet komennolla

```bash
poetry install
```

Käynnistä sovellus komennolla

```bash
poetry run invoke start
```

Suorita testit komennolla

```bash
poetry run invoke test
```

Käynnistä sovellus suorittaen ensin testit komennolla
```bash
poetry run invoke devstart
```

Aja testikattavuusraportti komennolla

```bash
poetry run invoke coverage
```

Tee Pylint-tarkastukset komennolla

```bash
poetry run invoke lint
```
## Minkä muotoisia syötteitä ohjelma hyväksyy
### Hyperparametrit:
- Käyttöliittymässä on olemassa oletuksena valmiit hyperparametrit, mutta niitä voi myös halutessaan muuttaa.

## Yleiskatsaus ohjelman käytöstä:
Käynnistyksen jälkeen ohjelman tulisi näyttää komentorivillä seuraavalta:

![image](https://github.com/user-attachments/assets/34cd3f1b-22ec-4ce4-b254-dd4cf0bc1d33)

Tällä hetkellä ohjelma tukee vain valintoja 1 ja 4.

Valitsemalla vaihtoehdon 1: Train network, pääset syöttämään verkon piilokerroksen koot eli neuronien määrän sekä muut hyperparametrit.

![image](https://github.com/user-attachments/assets/b26fe0c4-c781-49e5-bfcb-379f312b66ba)


