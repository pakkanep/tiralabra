## Ohjelman yleisrakenne: Kolmikerroksinen neuroverkko käsinkirjoitettujen numeroiden tunnistukseen

## Yleiskatsaus

- Tämä projekti toteuttaa **kolmikerroksisen neuroverkon**, jonka tehtävänä on tunnistaa käsinkirjoitettuja numeroita.
- Neuroverkko koostuu syötekerroksesta, yhdestä piilotetusta kerroksesta ja tuloskerroksesta.
- Opetuksessa käytetään **vastavirta (backpropagation)** algoritmia ja minisatsi gradienttimenetelmää sekä virhefunktiona **sigmoid-aktivointifunktiota**.

## Ohjelman rakenne

Ohjelma on rakennettu noudattamaan kerroksisen eteenpäin kytketyn neuroverkon periaatteita, ja se voidaan jakaa kolmeen päävaiheeseen:

### 1. Datan esikäsittely
- MNIST-tietokanta sisältää 70 000 mustavalkoista kuvaa käsinkirjoitetuista numeroista (0–9).
- Jokainen kuva on kooltaan 28x28 pikseliä, ja kuvat muunnetaan vektorimuotoon neuroverkon syötteitä varten.
- Ohjelma lataa MNIST-kuvat ja muokkaa ne 784-kokoisiksi syötevektoreiksi (28x28 = 784), jotka voidaan syöttää neuroverkon syötekerrokseen.

### 2. Neuroverkon arkkitehtuuri
Neuroverkko koostuu kolmesta kerroksesta:

- **Syötekerros**: 
  - Syötekerros vastaanottaa MNIST-kuvien syötteet. Koska kuvat ovat 28x28 pikseliä, syötekerroksessa on 784 neuronia (yksi jokaista pikseliä kohden).

- **Piilotettu kerros**: 
  - Neuroverkossa on yksi piilotettu kerros, joka sisältää esimerkiksi 128 neuronia. Tämä kerros prosessoi syötteen ja mahdollistaa verkon kyvyn oppia monimutkaisia kuvioita käsinkirjoitetuista numeroista.
  - Jokaisen piilotetun kerroksen neuronin aktivaatio lasketaan käyttämällä **sigmoid-funktiota**:
    ```math
    \sigma(x) = \frac{1}{1 + e^{-x}}
    ```

- **Tuloskerros**: 
  - Tuloskerroksessa on 10 neuronia, yksi jokaiselle numeroarvolle (0–9). Kunkin neuronin arvo edustaa todennäköisyyttä siitä, että syötteeksi annettu kuva vastaa kyseistä numeroa.

### 3. Verkon opettaminen
- Ensin syötetään minisatsin A kaikki opetusesimerkit x verkolle
- Kaikille opetusesimerkeille x:
  -  Lasketaan vastavirta-algoritmissa tarvittavat neuronikohtaiset summat z^l_j ja ulostulot a^l_{j} (Feedforward)
  -  Lasketaan syötettä vastaavan virhefunktion osittaisderivaatat vastavirta-algoritmin avulla (Backpropagation)
- päivitetään neuronien parametrit gradienttimenetelmän avulla.
```math
w^l \sim w^l - \frac{\alpha}{N} \sum_{x \in A} \delta_x^l (a_x^{l-1})^T
```

```math
b^l \sim b^l - \frac{\alpha}{N} \sum_{x \in A} \delta_x^l
```
missä alpha on verkon oppimisnopeus ja N minisatsin koko.

### Hyperparametrit
Hyperparametrit, kuten oppimisnopeus (learning rate), minisatsin koko (mini-batch size) ja epochien (koko harjoitusdatan läpikäynti) määrä vaikuttavat siihen kuinka nopeasti ja tarkasti neuroverkko oppii tunnistamaan numeroita.

## Saavutetut aika- ja tilavaativuudet (esim. O-analyysit pseudokoodista)


## Työn mahdolliset puutteet ja parannusehdotukset


## Laajojen kielimallien käyttö projektissa
    - En ole käyttänyt chat gpt:tä koodin generoimiseen.

### Olen käyttänyt chat-gpt:tä apuna seuraavissa ongelmissa:

#### Poetry ja riippuvuudet
    - Tensorflown lisääminen riippuvuudeksi epäonnistui, koska tensorflow ei toimi uusimman numpy version kanssa. Chat-gpt kertoi miksi ja että kannattaako numpya downgradeta

    - Kysyin Chatilta neuvoa vscoden konfigurointiin, kun koneeltani ei löytänyt tensorflow kirjastoa.

#### Testaus
    - Gradient check testiä varten vastavirta-algoritmin debuggaus.

    - Testejä varten tarkoituksena oli ladata MNist data tensorflowilla, mutta poetryn asentaessa kirjastoa, huomasin että kirjasto
    vaikuttaa turhan raskaalta vain tähän tarkotukseen. Seuraava ajatus oli vain ladata tensorflow.keras.datasets, jolloin chat-gpt ehdotti
    tensorflow-cpu:n lisäystä. Tämä osottautui turhaksi koska se oli liian hidasta.

    - Päädyin muokkaamaan itse testausta varten oman tiedoston, jotta isoa tiedostoa ei joka testauskerralla tarvitse purkaa.
    Tähän pyysin neuvoa chat-gpt:ltä. Vastaukseksi sain funktion joka loi .npz päätteisen tiedoston.


## Viitteet
https://tim.jyu.fi/view/143092#DKUvbnUuGytQ
http://neuralnetworksanddeeplearning.com/chap1.html
https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/
https://www.3blue1brown.com/topics/neural-networks
