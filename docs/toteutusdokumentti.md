## Laajojen kielimallien käyttö projektissa

### Olen käyttänyt chat-gpt:tä apuna seuraavissa ongelmissa:

#### Poetry ja riippuvuudet
    - Tensorflown lisääminen riippuvuudeksi epäonnistui, koska tensorflow ei toimi uusimman numpy version kanssa. Chat-gpt kertoi miksi ja että kannattaako numpya downgradeta

    - Kysyin Chatilta neuvoa vscoden konfigurointiin, kun koneeltani ei löytänyt tensorflow kirjastoa.

#### Testaus

    - Testejä varten tarkoituksena oli ladata MNist data tensorflowilla, mutta poetryn asentaessa kirjastoa, huomasin että kirjasto
    vaikuttaa turhan raskaalta vain tähän tarkotukseen. Seuraava ajatus oli vain ladata tensorflow.keras.datasets, jolloin chat-gpt ehdotti
    tensorflow-cpu:n lisäystä. Tämä osottautui turhaksi koska se oli liian hidasta.

    - Päädyin muokkaamaan itse testausta varten oman tiedoston, jotta isoa tiedostoa ei joka testauskerralla tarvitse purkaa.
    Tähän pyysin neuvoa chat-gpt:ltä. Vastaukseksi sain funktion joka loi .npz päätteisen tiedoston.


