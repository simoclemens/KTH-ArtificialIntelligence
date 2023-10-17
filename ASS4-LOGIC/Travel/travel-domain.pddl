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
  
  ; Person x enters vehicle y if both are in the same airport/station z.
  ; As a result, person x is in vehicle y and not at z anymore.
  ; Parameters
  ; - x: person
  ; - y: vehicle
  ; - z: station
  (:action enter-vehicle
    ; Complete code here
  )
  
  ; Person x leaves vehicle y in airport/station z if the person x is in the 
  ; vehicle y and the vehicle y is at z.
  ; As a result, person x is not in vehicle y anymore and the person x is at z
  ; Parameters
  ; - x: person
  ; - y: vehicle
  ; - z: station
  (:action leave-vehicle
    ; Complete code here
  )

  ; Long-distance travel, i.e. between airports x and y by an 
  ; airplane z if x and y are connected.
  ; As a result, vehicle z is at y, and not at x anymore.
  ; Parameters
  ; - x: airport from
  ; - y: airport to
  ; - z: airplane
  (:action travel-long
    ; Complete code here
  )

  ; Short-distance travel, i.e. not between airports, by a 
  ; subway train z if x and y are connected.
  ; As a result, vehicle z is at y, and not at x anymore.
  ; Parameters
  ; - x: airport/station from
  ; - y: airport/station to
  ; - z: subway
  (:action travel-short
    ; Complete code here
  )
  
  
)