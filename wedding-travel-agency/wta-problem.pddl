(define (problem wta_explore_bayern_from_rome)
  (:domain wta)
  (:objects
    Bamberg Nurnberg Regensburg Munchen Lindau Bregenz
    Fussen Innsbruck Rothenburg Wurzburg Rome Bologna
    Ettal Linderhof
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

    RomeMunich
    - air_travel

    RomeBologna BolognaInnsbruck BolognaMunich
    - train_travel

    
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
    (hotel_at HotelRothenburgerHof Rothenburger)
    (hotel_at HotelStraussWurzburg Wurzburg)
    (hotel_at KlosterhotelEttal Ettal)
    (hotel_at HotelPalaceBologna Bologna)
    (attraction_at NeuschwansteinCastle Fussen)
    (attraction_at EttalAbbey Ettal)
    (attraction_at SchlossLinderhof Linderhof)
    (attraction_at LindauHafen Lindau)
    (attraction_at HohesSchloss Fussen)
    (attraction_at Pfander Bregenz)
    (attraction_at RothenburgObDerTauber Rothenburg)
    (attraction_at WurzburgResidence Wurzburg)
    (attraction_at BambergOldTown Bamberg)
    (attraction_at ImperialCastleOfNuremberg Nuremberg)
    (attraction_at AltesRathausRegensburg Regensburg)
    (attraction_at MunichResidence Munich)
    (attraction_at SchlossNymphenburg Munich)
    (attraction_at PiazzaMaggiore Bologna)
    (connect RomeMunich Rome Munich)
    (connect RomeBologna Rome Bologna)
    (connect BolognaInnsbruck Bologna Innsbruck)
    (connect BolognaMunich Bologna Munich)
  )
  (:goal (and 
    (at Rome) ;; ending at home
    (<= (total-cost) 2000) ;; max budget
    (<= (days) 10) ;; in 10 days
  ))
  (:metric maximise attractions_visited)
)