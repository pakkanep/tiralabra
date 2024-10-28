# Testaus
### Testisyötteet:
- Testisyötteenä kaikissa testeissä toimii MNist data eli sama data kun millä verkkoa opetetaan ja testataan. Kuitenkin niin että käyetään vain osaa koko datasta ja osa valitaan aina satunnaisesti.

### NeuralNet luokka:
Testataan että:
- luokittelutarkkuuden laskeva metodi toimii silloin kun syöte-tavoite parissa x, y, y pitää muuntaa.
- luokittelutarkkuuden laskeva metodi toimii silloin kun syöte-tavoite parissa x, y, y ei tarvitse muuntaa.
- verkon tuloksen palauttava metodi toimii, koska tätä käytetään monessa muussa metodisssa.
  
### Koko verkon toiminta:
Testataan, että:
- Verkon kyky tunnistaa numeroita paranee yhden erän jälkeen opettamalla verkkoa normaalisti mutta vain pienemmällä piilokerroksella sekä pienemällä määrällä dataa.
- Virhefunktion arvo pienenee kahden opetuserän jälkeen, mutta osittaisella datalla.
- Kaikki verkon piilokerrokset muuttuvat kierrosten välissä. Varmistaa että kaikki kerrokset ovat käytössä.
- syöte-tavoite parien järjestys erässä (batch) ei vaikuta verkon tulokseen.
- stokastinen gradienttifunktio menee lokaalia minimiä kohti

### Vastavirta algoritmi:  
- Vastavirta algoritmin palauttamat gradientti vektorit muuttuvat eli, eivät sisällä pelkkiä nollia.
- Vastavirta algoritmin laskemat gradientit ovat oikein laskettu. Tämä testataan gradient checkillä, jossa siis lasketaan numeerinen gradientti ja verrataan sitä analyyttiseen gradienttiin.

### Poissuljettu coveragesta ja miksi:
- src/tests/* ei olennainen
  
- exclude_lines=
  
    def print_progress

    self.print_progress

poistettu koska printtejä ei varsinaisesti voi testata


![image](https://github.com/user-attachments/assets/beca116a-be40-4569-80b2-3f310f328fc8)


![image](https://github.com/user-attachments/assets/24390594-bc2a-4725-a520-c4024f13866c)

