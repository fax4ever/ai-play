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
    (at Rome) ;; starting from home
    (hotel_at HotelEuropaBamberg Bamberg)
    (hotel_at ParkInnNurnberg Nurnberg)
    (hotel_at IbisStylesRegensburg Regensburg)
    (hotel_at BoutiqueHotelMunchen Munchen)
    (hotel_at HotelEngelLindau Lindau)
    (hotel_at IbisBregenz Bregenz)
    (hotel_at HotelSchlosskroneFussen Fussen)
    (hotel_at YouthHostelInnsbruck Innsbruck)
    (hotel_at HotelRothenburgerHof Rothenburg)
    (hotel_at HotelStraussWurzburg Wurzburg)
    (hotel_at KlosterhotelEttal Ettal)
    (hotel_at HotelPalaceBologna Bologna)
    (attraction_at NeuschwansteinCastle Fussen)
    (attraction_at EttalAbbey Ettal)
    (attraction_at SchlossLinderhof Ettal)
    (attraction_at LindauHafen Lindau)
    (attraction_at HohesSchloss Fussen)
    (attraction_at Pfander Bregenz)
    (attraction_at RothenburgObDerTauber Rothenburg)
    (attraction_at WurzburgResidence Wurzburg)
    (attraction_at BambergOldTown Bamberg)
    (attraction_at ImperialCastleOfNuremberg Nurnberg)
    (attraction_at AltesRathausRegensburg Regensburg)
    (attraction_at MunichResidence Munchen)
    (attraction_at SchlossNymphenburg Munchen)
    (attraction_at PiazzaMaggiore Bologna)
    (connect RomeMunchen Rome Munchen)
    (= (time_travel RomeMunchen) 240)
    (connect RomeBologna Rome Bologna)
    (connect BolognaInnsbruck Bologna Innsbruck)
    (connect BolognaMunchen Bologna Munchen)
    (connect WurzburgBamberg Wurzburg Bamberg)
    (connect WurzburgRothenburg Wurzburg Rothenburg)
    (connect BambergRothenburg Bamberg Rothenburg)
    (connect BambergNurnberg Bamberg Nurnberg)
    (connect NurnbergRothenburg Nurnberg Rothenburg)
    (connect NurnbergRegensburg Nurnberg Regensburg)
    (connect NurnbergMunchen Nurnberg Munchen)
    (connect RothenburgMunchen Rothenburg Munchen)
    (connect RothenburgFussen Rothenburg Fussen)
    (connect RothenburgLindau Rothenburg Lindau)
    (connect LindauMunchen Lindau Munchen)
    (connect LindauFussen Lindau Fussen)
    (connect LindauBregenz Lindau Bregenz)
    (connect BregenzInnsbruck Bregenz Innsbruck)
    (connect FussenEttal Fussen Ettal)
    (connect EttalMunchen Ettal Munchen)
    (connect EttalInnsbruck Ettal Innsbruck)
    (connect MunchenInnsbruck Munchen Innsbruck)
    (= (day_hours) 0)
    (= (max_day_hours) 10)
    (= (days) 0)
    (= (attractions_visited) 0)
  )
  (:goal (and 
    (at Rome) ;; ending at home
    (<= (total-cost) 2000) ;; max budget
    (<= (days) 10) ;; in 10 days
  ))
  (:metric maximize (attractions_visited))
)