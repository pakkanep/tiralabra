# tiralabra

Neural network for recognizing handwritten digits

HY - Aineopintojen harjoitustyö: Algoritmit ja tekoäly

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
