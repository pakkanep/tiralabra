
#### Mitä olen tehnyt tällä viikolla?
- Koitin kirjoittaa gradient check testiä ja debuggata backprop algoa.
- Muutin verkon painojen alustamisen, joka nopeutti verkon oppimista jonkin verran.
  
#### Miten ohjelma on edistynyt?
- Sain muokattua vastavirta algon ja minisatsin laskemaan yhden erän kaikki gradienttivektorit matriisisissa rinnakain, joka nopeutti laskentaa huomattavasti.

#### Mitä opin tällä viikolla / tänään?
- Gradient check menetelmästä
  
#### Mikä jäi epäselväksi tai tuottanut vaikeuksia?
- Epäselväksi jäi mikä bugi gradient check testissä tai vastavirta algossa on. Kuitenkin 100 neuronin piilokerroksella sain verkon tarkkuuddeksi 97%. Mainittakoon myös, että painojen numeeristen ja analyyttisten gradienttien erotukset ovat luokkaa 10<sup>-10</sup>. 


#### Mitä teen seuraavaksi?
- Kirjoitan testit loppuun.
- Kirjoitan docstingit funktioille.
- Teen testidokumentin valmiiksi.
   



## Tuntikirjanpito

| Päivä | Käytetty aika | Kuvaus |
| ----- | ------------- | ------ |
| 23.9.  | 3h            | Teorian lukeminen |
| 24.9.  | 5h            | Vastavirta algoritmin debuggaus gradient check testin takia |
| 25.9.  | 3h            | Vastavirta algoritmin debuggaus gradient check testin takia ja pieni muutos luokan konstruktoriin |
| 26.9.  | 8h            | Testien kirjoittelu, minisatsin ja vastavirta funktion muokkaus, sekä pari muuta uutta funktiota Neural_Net luokkaan  |
| Yhteensä | 19h         |        |
