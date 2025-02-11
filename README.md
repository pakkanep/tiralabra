# Algorithms-and-Artificial-Intelligence-Project
![GHA workflow badge](https://github.com/pakkanep/tiralabra/workflows/CI/badge.svg)
[![codecov](https://codecov.io/github/pakkanep/tiralabra/graph/badge.svg?token=4FUGYDMMPR)](https://codecov.io/github/pakkanep/tiralabra)

Neuroverkko käsinkirjoitettujen numeroiden tunnistamiseen / Neural network for recognizing handwritten digits

Helsingin Yliopisto - Aineopintojen harjoitustyö: Algoritmit ja tekoäly / University of Helsinki Bachelor's Level Course Project: Algorithms and Artificial Intelligence

## Dokumentaatio
[Määrittelydokumentti](./docs/maarittelydokumentti.md)

[Testausdokumentti](./docs/testausdokumentti.md)

[Toteutusdokumentti](./docs/toteutusdokumentti.md)

[Käyttöohje](./docs/kayttoohje.md)

## Viikkoraportit
[Viikko 1](./docs/viikkoraportit/viikkoraportti1.md)

[Viikko 2](./docs/viikkoraportit/viikkoraportti2.md)

[Viikko 3](./docs/viikkoraportit/viikkoraportti3.md)

[Viikko 4](./docs/viikkoraportit/viikkoraportti4.md)

[Viikko 5](./docs/viikkoraportit/viikkoraportti5.md)

[Viikko 6](./docs/viikkoraportit/viikkoraportti6.md)

## Asennus ja käyttö / Installation and Usage
Asenna riippuvuudet komennolla / Install dependencies with the command

```bash
poetry install
```

Käynnistä sovellus komennolla / Start the application with the command

```bash
poetry run invoke start
```

Suorita testit komennolla / Run the tests

```bash
poetry run invoke test
```

Käynnistä sovellus suorittaen ensin testit komennolla / Run tests and then start the application
```bash
poetry run invoke devstart
```

Aja testikattavuusraportti komennolla / Run the test coverage report with the command

```bash
poetry run invoke coverage
```

Tee Pylint-tarkastukset komennolla / Perform Pylint checks with the command

```bash
poetry run invoke lint
```
