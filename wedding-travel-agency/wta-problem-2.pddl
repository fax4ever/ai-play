(define (problem wta_explore_bayern_from_rome)
  (:domain wta)
  (:objects
    Bamberg Nurnberg Regensburg Munchen Lindau Bregenz
    Fussen Innsbruck Rothenburg Wurzburg Rome Bologna Ettal
    - place

    HotelEuropaBamberg ParkInnNurnberg IbisStylesRegensburg
    BoutiqueHotelMunchen HotelEngelLindau IbisBregenz
    HotelSchlosskroneFussen YouthHostelInnsbruck
    HotelRothenburgerHof HotelStraussWurzburg
    KlosterhotelEttal HotelPalaceBologna
    - hotel

    NeuschwansteinCastle EttalAbbey SchlossLinderhof
    LindauHafen HohesSchloss Pfander RothenburgObDerTauber
    WurzburgResidence BambergOldTown ImperialCastleOfNuremberg
    AltesRathausRegensburg MunichResidence SchlossNymphenburg
    PiazzaMaggiore
    - attraction

    RomeMunchen
    - air_travel

    RomeBologna BolognaInnsbruck BolognaMunchen
    - train_travel

    WurzburgBamberg WurzburgRothenburg BambergRothenburg BambergNurnberg 
    NurnbergRothenburg NurnbergRegensburg NurnbergMunchen
    RothenburgMunchen RothenburgFussen RothenburgLindau
    LindauMunchen LindauFussen LindauBregenz
    BregenzInnsbruck FussenEttal EttalMunchen EttalInnsbruck
    MunchenInnsbruck
    - car_travel
  )
  (:init
    (at Bamberg) ;; starting from Bamberg
    (hotel_at HotelEuropaBamberg Bamberg)
    (= (cost_hotel HotelEuropaBamberg) 89)
    (hotel_at ParkInnNurnberg Nurnberg)
    (= (cost_hotel ParkInnNurnberg) 80)
    (hotel_at IbisStylesRegensburg Regensburg)
    (= (cost_hotel IbisStylesRegensburg) 138)
    (hotel_at BoutiqueHotelMunchen Munchen)
    (= (cost_hotel BoutiqueHotelMunchen) 170)
    (hotel_at HotelEngelLindau Lindau)
    (= (cost_hotel HotelEngelLindau) 150)
    (hotel_at IbisBregenz Bregenz)
    (= (cost_hotel IbisBregenz) 94)
    (hotel_at HotelSchlosskroneFussen Fussen)
    (= (cost_hotel HotelSchlosskroneFussen) 134)
    (hotel_at YouthHostelInnsbruck Innsbruck)
    (= (cost_hotel YouthHostelInnsbruck) 88)
    (hotel_at HotelRothenburgerHof Rothenburg)
    (= (cost_hotel HotelRothenburgerHof) 70)
    (hotel_at HotelStraussWurzburg Wurzburg)
    (= (cost_hotel HotelStraussWurzburg) 89)
    (hotel_at KlosterhotelEttal Ettal)
    (= (cost_hotel KlosterhotelEttal) 143)
    (hotel_at HotelPalaceBologna Bologna)
    (= (cost_hotel HotelPalaceBologna) 115)
    (attraction_at NeuschwansteinCastle Fussen)
    (= (time_attraction NeuschwansteinCastle) 240)
    (= (cost_attraction NeuschwansteinCastle) 21)
    (attraction_at EttalAbbey Ettal)
    (= (time_attraction EttalAbbey) 60)
    (= (cost_attraction EttalAbbey) 0)
    (attraction_at SchlossLinderhof Ettal)
    (= (time_attraction SchlossLinderhof) 120)
    (= (cost_attraction SchlossLinderhof) 10)
    (attraction_at LindauHafen Lindau)
    (= (time_attraction LindauHafen) 120)
    (= (cost_attraction LindauHafen) 0)
    (attraction_at HohesSchloss Fussen)
    (= (time_attraction HohesSchloss) 60)
    (= (cost_attraction HohesSchloss) 5)
    (attraction_at Pfander Bregenz)
    (= (time_attraction Pfander) 60)
    (= (cost_attraction Pfander) 10)
    (attraction_at RothenburgObDerTauber Rothenburg)
    (= (time_attraction RothenburgObDerTauber) 120)
    (= (cost_attraction RothenburgObDerTauber) 0)
    (attraction_at WurzburgResidence Wurzburg)
    (= (time_attraction WurzburgResidence) 120)
    (= (cost_attraction WurzburgResidence) 10)
    (attraction_at BambergOldTown Bamberg)
    (= (time_attraction BambergOldTown) 120)
    (= (cost_attraction BambergOldTown) 0)
    (attraction_at ImperialCastleOfNuremberg Nurnberg)
    (= (time_attraction ImperialCastleOfNuremberg) 60)
    (= (cost_attraction ImperialCastleOfNuremberg) 7)
    (attraction_at AltesRathausRegensburg Regensburg)
    (= (time_attraction AltesRathausRegensburg) 30)
    (= (cost_attraction AltesRathausRegensburg) 0)
    (attraction_at MunichResidence Munchen)
    (= (time_attraction MunichResidence) 90)
    (= (cost_attraction MunichResidence) 10)
    (attraction_at SchlossNymphenburg Munchen)
    (= (time_attraction SchlossNymphenburg) 120)
    (= (cost_attraction SchlossNymphenburg) 10)
    (attraction_at PiazzaMaggiore Bologna)
    (= (time_attraction PiazzaMaggiore) 20)
    (= (cost_attraction PiazzaMaggiore) 0)
    (connect RomeMunchen Rome Munchen)
    (= (time_travel RomeMunchen) 120)
    (= (cost_travel RomeMunchen) 200)
    (connect RomeBologna Rome Bologna)
    (= (time_travel RomeBologna) 120)
    (= (cost_travel RomeBologna) 67)
    (connect BolognaInnsbruck Bologna Innsbruck)
    (= (time_travel BolognaInnsbruck) 300)
    (= (cost_travel BolognaInnsbruck) 45)
    (connect BolognaMunchen Bologna Munchen)
    (= (time_travel BolognaMunchen) 420)
    (= (cost_travel BolognaInnsbruck) 70)
    (connect WurzburgBamberg Wurzburg Bamberg)
    (= (time_travel WurzburgBamberg) 60)
    (= (cost_travel WurzburgBamberg) 6)
    (connect WurzburgRothenburg Wurzburg Rothenburg)
    (= (time_travel WurzburgRothenburg) 41)
    (= (cost_travel WurzburgRothenburg) 4)
    (connect BambergRothenburg Bamberg Rothenburg)
    (= (time_travel BambergRothenburg) 89)
    (= (cost_travel BambergRothenburg) 9)
    (connect BambergNurnberg Bamberg Nurnberg)
    (= (time_travel BambergNurnberg) 52)
    (= (cost_travel BambergNurnberg) 5)
    (connect NurnbergRothenburg Nurnberg Rothenburg)
    (= (time_travel NurnbergRothenburg) 75)
    (= (cost_travel NurnbergRothenburg) 7)
    (connect NurnbergRegensburg Nurnberg Regensburg)
    (= (time_travel NurnbergRegensburg) 79)
    (= (cost_travel NurnbergRegensburg) 8)
    (connect NurnbergMunchen Nurnberg Munchen)
    (= (time_travel NurnbergMunchen) 121)
    (= (cost_travel NurnbergMunchen) 12)
    (connect RothenburgMunchen Rothenburg Munchen)
    (= (time_travel RothenburgMunchen) 167)
    (= (cost_travel RothenburgMunchen) 17)
    (connect RothenburgFussen Rothenburg Fussen)
    (= (time_travel RothenburgFussen) 143)
    (= (cost_travel RothenburgFussen) 14)
    (connect RothenburgLindau Rothenburg Lindau)
    (= (time_travel RothenburgLindau) 142)
    (= (cost_travel RothenburgLindau) 14)
    (connect LindauMunchen Lindau Munchen)
    (= (time_travel LindauMunchen) 128)
    (= (cost_travel LindauMunchen) 13)
    (connect LindauFussen Lindau Fussen)
    (= (time_travel LindauFussen) 77)
    (= (cost_travel LindauFussen) 8)
    (connect LindauBregenz Lindau Bregenz)
    (= (time_travel LindauBregenz) 16)
    (= (cost_travel LindauBregenz) 2)
    (connect BregenzInnsbruck Bregenz Innsbruck)
    (= (time_travel BregenzInnsbruck) 129)
    (= (cost_travel BregenzInnsbruck) 13)
    (connect FussenEttal Fussen Ettal)
    (= (time_travel FussenEttal) 46)
    (= (cost_travel FussenEttal) 5)
    (connect EttalMunchen Ettal Munchen)
    (= (time_travel EttalMunchen) 66)
    (= (cost_travel EttalMunchen) 7)
    (connect EttalInnsbruck Ettal Innsbruck)
    (= (time_travel EttalInnsbruck) 69)
    (= (cost_travel EttalInnsbruck) 7)
    (connect MunchenInnsbruck Munchen Innsbruck)
    (= (time_travel MunchenInnsbruck) 121)
    (= (cost_travel MunchenInnsbruck) 12)
    (= (day_activity_minutes) 0)
    (= (max_day_activity_minutes) 600)
    (= (days) 0)
    (= (attractions_visited) 0)
  )
  (:goal (and 
    (at Ettal) ;; ending at Ettal
    (visited EttalAbbey)
    (visited BambergOldTown)
    (visited WurzburgResidence)
    (visited AltesRathausRegensburg)
  ))
  ;; minimizing the total cost
  ;; getting a bonus for each visisted attractions
  (:metric minimize (- (total-cost) (* 100 (attractions_visited))))
)