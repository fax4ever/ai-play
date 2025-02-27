(define (domain wta)
  (:requirements :typing :fluents :disjunctive-preconditions :negative-preconditions)
  (:types
    place attraction hotel travel - object
    train_travel air_travel car_travel - travel
  )
  (:predicates
    (at ?p - place)
    (connect ?t - travel ?from ?to - place)
    (hotel_at ?h - hotel ?p - place)
    (attraction_at ?a - attraction ?p - place)
    (visited ?a - attraction)
  )
  (:functions
    (time_travel ?t - travel)
    (time_attraction ?a - attraction)
    (cost_travel ?t - travel)
    (cost_attraction ?a - attraction)
    (cost_hotel ?h - hotel)
    ;; `day_activity_minutes`: denotes the current day activities time
    ;; in minutes
    ;; travelling and visiting the attractions
    (day_activity_minutes)
    ;; this value cannot exceed `max_day_activity_minutes`
    (max_day_activity_minutes)
    ;; `days`: the days have passed so far
    (days)
    (attractions_visited)
    (total-cost)
  )
  (:action move
    :parameters (?t - travel ?from ?to - place)
    :precondition (and 
      (at ?from) 
      (or (connect ?t ?from ?to) (connect ?t ?to ?from))
      (<= (+ (day_activity_minutes) (time_travel ?t)) (max_day_activity_minutes))
    )
    :effect (and
      (increase (total-cost) (cost_travel ?t))
      (increase (day_activity_minutes) (time_travel ?t))
      (not (at ?from))
      (at ?to)
    )
  )
  (:action visit
    :parameters (?a - attraction ?p - place)
    :precondition (and
      ;; Each attraction can be visited once
      (not (visited ?a))
      (at ?p)
      (attraction_at ?a ?p)
      (<= (+ (day_activity_minutes) (time_attraction ?a)) (max_day_activity_minutes))
    )
    :effect (and
      (increase (total-cost) (cost_attraction ?a))
      (increase (day_activity_minutes) (time_attraction ?a))
      (visited ?a)
      (increase (attractions_visited) 1)
    )
  )
  (:action rest
    :parameters (?h - hotel ?p - place)
    :precondition (and
      ;; Hotels can be visited more than once
      (at ?p)
      (hotel_at ?h ?p)
    )
    :effect (and
      (increase (total-cost) (cost_hotel ?h))
      ;; `day_activity_minutes` is reset to 0
      ;; so that tomorrow I can visit more attractions
      (assign (day_activity_minutes) 0)
      ;; one day is added
      (increase (days) 1)
    )
  )
)