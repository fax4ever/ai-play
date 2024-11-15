(define (problem termes_1) (:domain termes)
(:objects 
    n0 n1 n2 n3 - numb
    pos-0-0 pos-0-1 pos-0-2 - position
)

(:init
    (height pos-0-0 n0)
    (NEIGHBOR pos-0-0 pos-0-1)
)

(:goal (and
    (height pos-0-0 n0)
    (not (has-block))
))
)
