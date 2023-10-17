;; Domain definition
(define (domain travel-domain)
  
  ;; Predicates: Properties of objects that we are interested in (boolean)
  (:predicates
    (AIRPORT ?x) ; True if x is an airport
    (STATION ?x) ; True if x is a station
    (PERSON ?x) ; True if x is a person
    (VEHICLE ?x) ; True if x is a method of transportation
    (AIRPLANE ?x) ; True if x is an airplane
    (SUBWAY ?x) ; True if x is a subway
    (connected ?x ?y) ; True if airport/station x is connected to airport/station y
    (is-person-at ?x ?y) ; True if person x is at airport/station y
    (is-vehicle-at ?x ?y) ; True if vehicle x is at airport/station y
    (is-person-in-vehicle ?x ?y) ; True if person x is in vehicle y
  )

  ;; Actions: Ways of changing the state of the world
  
  (:action enter-vehicle
    :parameters (?x - PERSON ?y - VEHICLE ?z - STATION)
    :precondition (and 
                    (is-person-at ?x ?z)
                    (is-vehicle-at ?y ?z)
                    (not (is-person-in-vehicle ?x ?y)))
    :effect (and 
              (is-person-in-vehicle ?x ?y)
              (not (is-person-at ?x ?z)))
  )
  
  (:action leave-vehicle
    :parameters (?x - PERSON ?y - VEHICLE ?z - STATION)
    :precondition (and 
                    (is-person-in-vehicle ?x ?y)
                    (is-vehicle-at ?y ?z))
    :effect (and 
              (is-person-at ?x ?z)
              (not (is-person-in-vehicle ?x ?y)))
  )

  (:action travel-long
    :parameters (?x - AIRPORT ?y - AIRPORT ?z - AIRPLANE)
    :precondition (and 
                    (connected ?x ?y)
                    (is-vehicle-at ?z ?x))
    :effect (and 
              (is-vehicle-at ?z ?y)
              (not (is-vehicle-at ?z ?x)))
  )

  (:action travel-short
    :parameters (?x - STATION ?y - STATION ?z - SUBWAY)
    :precondition (and 
                    (connected ?x ?y)
                    (is-vehicle-at ?z ?x))
    :effect (and 
              (is-vehicle-at ?z ?y)
              (not (is-vehicle-at ?z ?x)))
  )
  
)
