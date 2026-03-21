# SPARQL Queries

## SPARQL 1.0

### List all classes
```sparql
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT DISTINCT ?class
WHERE {
    ?class rdf:type owl:Class .
}
ORDER BY ?class
```
**Returns**: All declared classes in the ontology

### Query: List all object properties
```sparql
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT DISTINCT ?property
WHERE {
    ?property rdf:type owl:ObjectProperty .
}
ORDER BY ?property
```
**Returns**: All object properties

### Query: List all data properties
```sparql
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT DISTINCT ?property
WHERE {
    ?property rdf:type owl:DatatypeProperty .
}
ORDER BY ?property
```
**Returns**: All data properties

### Query: Find all Persons (explicit only)
```sparql
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT ?person
WHERE {
    ?person rdf:type :Person .
}
```
**Returns**: Only individuals explicitly asserted as Person
**Compare with DL Query**: `Person` returns MORE (includes inferred Employees, Managers, etc.)
Try also: *Person > Employee > Manager > Executive > CEO*

### Query: Types of IRIs

```sparql
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT DISTINCT ?iri ?type
WHERE {
    ?iri rdf:type ?type .
    FILTER (!isBlank(?iri))
}
ORDER BY ?iri
```

### Query: Who supervises whom?

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT ?supervisor ?employee
WHERE {
    ?supervisor :supervises ?employee .
}
```

### Query: Who works in which department?

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT ?person ?department
WHERE {
    ?person :worksIn ?department .
}
```

### Query: Who works on which project?

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT ?person ?project
WHERE {
    ?person :worksOn ?project .
}
```

### Query: Who manages what?

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT ?manager ?managed
WHERE {
    ?manager :manages ?managed .
}
```

### Query: Find colleague relationships

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT ?person1 ?person2
WHERE {
    ?person1 :colleagueOf ?person2 .
}
```

### Query: Find mentorship relationships

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT ?mentor ?mentee
WHERE {
    ?mentor :mentors ?mentee .
}
```

### Query: Employees / IDs / Names / Salaries

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT ?employee ?id ?name ?salary ?age
WHERE {
    OPTIONAL {?employee :hasSalary ?salary} .
    OPTIONAL {?employee :hasEmployeeID ?id} .
    OPTIONAL {?employee :hasName ?name} .
    OPTIONAL {?employee :hasAge ?age}
}
```

### Query: Employees Mid Range Salaries

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT ?employee ?name ?salary
WHERE {
    ?employee rdf:type :Employee .
    ?employee :hasName ?name .
    ?employee :hasSalary ?salary .
    FILTER (?salary >= 50000 && ?salary <= 100000)
}
ORDER BY ?salary
```

### Query: Employees with optional project assignment

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT ?employee ?name ?project
WHERE {
    ?employee rdf:type :Employee .
    ?employee :hasName ?name .
    OPTIONAL { ?employee :worksOn ?project }
}
```

### Query: Employees or Managers

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT DISTINCT ?worker ?name
WHERE {
    ?worker :hasName ?name .
    {
        ?worker rdf:type :Employee .
    }
    UNION
    {
        ?worker rdf:type :Manager .
    }
}
```

### Query: Supervised team size

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT ?manager ?name (COUNT(?supervised) as ?teamSize)
WHERE {
    ?manager rdf:type :Manager .
    ?manager :hasName ?name .
    ?manager :supervises ?supervised .
}
GROUP BY ?manager ?name
ORDER BY DESC(?teamSize)
```

### Query: Salary statistics

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT (AVG(?salary) as ?avgSalary) (MIN(?salary) as ?minSalary) (MAX(?salary) as ?maxSalary)
WHERE {
    ?employee :hasSalary ?salary .
}
```

### Query: Age statistics

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT (AVG(?age) as ?avgAge) (MIN(?age) as ?minAge) (MAX(?age) as ?maxAge)
WHERE {
    ?employee :hasAge ?age .
}
```

## SPARQL 1.1

### Transitive Query: Transitive supervision

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT ?supervisor ?subordinate
WHERE {
    ?supervisor :supervises+ ?subordinate .
}
```

### Transitive Query: All people supervised by JohnSmith (direct and indirect)

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT ?subordinate ?name
WHERE {
    :JohnSmith :supervises+ ?subordinate .
    ?subordinate :hasName ?name .
}
```

### Transitive Query: All individuals of Person (including subclasses)
(Simulates reasoning)

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?individual ?directType ?name
WHERE {
    ?individual rdf:type ?directType .
    ?directType rdfs:subClassOf* :Person .
    OPTIONAL { ?individual :hasName ?name }
}
ORDER BY ?name
```

### Transitive Query: All individuals of Employee (including subclasses)
(Simulates reasoning)

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?individual ?actualClass ?name ?salary
WHERE {
    ?individual rdf:type ?actualClass .
    ?actualClass rdfs:subClassOf* :Employee .
    OPTIONAL { ?individual :hasName ?name }
    OPTIONAL { ?individual :hasSalary ?salary }
}
ORDER BY DESC(?salary)
```

### Transitive Query: Find all ancestors of a class
(Simulates reasoning)

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?ancestor
WHERE {
    :CEO rdfs:subClassOf+ ?ancestor .
    FILTER (!isBlank(?ancestor))
}
ORDER BY ?ancestor
```

### Transitive Query: All Employees (via hierarchy) with "Smith" in name

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?employee ?class ?name
WHERE {
    ?employee rdf:type ?class .
    ?class rdfs:subClassOf* :Employee .
    ?employee :hasName ?name .
    FILTER(REGEX(?name, "Smith", "i"))
}
```

### Transitive Query: Bidirectional property paths - Find all related people

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT DISTINCT ?person1 ?person2
WHERE {
    ?person1 (:supervises | ^:supervises)+ ?person2 .
    FILTER(?person1 != ?person2)
}
```

### Transitive Query: Sequence of property paths (/) - Department to building

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT ?person ?personName ?building 
WHERE {
    ?person :worksIn / :locatedIn ?building .
    ?person :hasName ?personName .
}
```

### Transitive Query: Complete family tree pattern

```sparql
PREFIX : <http://www.semanticweb.org/fax/ontologies/2026/corporate#>

SELECT DISTINCT ?subject ?ancestor
WHERE {
    {
        # Upward in hierarchy
        ?subject (:reportsTo | :supervisedBy | ^:supervises | ^:manages)+ ?ancestor .
    }
    UNION
    {
        # Downward in hierarchy
        ?ancestor (:supervises | :manages)+ ?subject .
    }

    Optional {?ancestor :hasName ?ancestorName} .
    FILTER(?ancestor != ?subject)
}
ORDER BY ?ancestorName
```