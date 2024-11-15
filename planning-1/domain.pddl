(define (domain termes)
(:requirements :strips :typing :negative-preconditions) 

(:types
    position - object
    numb - object
)

(:predicates
    (height ?p - position ?h - numb)
    (at ?p - position)
    (has-block)

    (NEIGHBOR ?p1 - position ?p2 - position)
    (IS-DEPOT ?p - position)
    (SUCC ?n1 - numb ?n2 - numb)
)

(:action move
    :parameters (?from ?to - position ?h - numb)
    :precondition (and 
        (at ?from)
        (NEIGHBOR ?from ?to)
        (height ?from ?h)
        (height ?to ?h)
    )
    :effect (and 
        (not (at ?from))
        (at ?to)
    )
)

(:action move-up
    :parameters (?from ?to - position ?hfrom ?hto - position)
    :precondition (and 
        (at ?from)
        (NEIGHBOR ?from ?to)
        (height ?from ?hfrom)
        (height ?to ?hto)
        (SUCC ?hto ?hfrom)
    )
    :effect (and 
        (not (at ?from))
        (at ?to)
    )
)

(:action move-down
    :parameters (?from ?to - position ?hfrom ?hto - position)
    :precondition (and 
        (at ?from)
        (NEIGHBOR ?from ?to)
        (height ?from ?hfrom)
        (height ?to ?hto)
        (SUCC ?hfrom ?hto)
    )
    :effect (and 
        (not (at ?from))
        (at ?to)
    )
)

(:action place-block
    :parameters (?rpos ?bpos - position ?hbefore ?hafter - numb)
    :precondition (and 
        (at ?rpos)
        (NEIGHBOR ?rpos ?bpos)
        (height ?rpos ?hbefore)
        (height ?bpos ?hbefore)
        (has-block)
        (not (IS-DEPOT ?bpos))
        (SUCC ?hafter ?hbefore)
    )
    :effect (and 
        (not (height ?bpos ?hbefore))
        (height ?bpos ?hafter)
    )
)

(:action remove-block
    :parameters (?rpos ?bpos - position ?hbefore ?hafter - numb)
    :precondition (and 
        (at ?rpos)
        (NEIGHBOR ?rpos ?bpos)
        (height ?rpos ?hafter)
        (height ?bpos ?hbefore)
        (not (has-block))
        (not (IS-DEPOT ?bpos))
        (SUCC ?hafter ?hbefore)
    )
    :effect (and 
        (not (height ?bpos ?hbefore))
        (height ?bpos ?hafter)
        (has-block)
    )
)

(:action create-block
    :parameters (?p - position)
    :precondition (and 
        (at ?p)
        (IS-DEPOT ?p)
        (not (has-block))
    )
    :effect (and 
        (has-block)
    )
)

(:action destroy-block
    :parameters (?p - position)
    :precondition (and 
        (at ?p)
        (IS-DEPOT ?p)
        (not (has-block))
    )
    :effect (and 
        (not (has-block))
    )
)


)