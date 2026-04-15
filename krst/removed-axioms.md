# Manual Changes

The intitial version of the ontology has been produced using Claude code. The result was good but it presented a series of issues.

## Removed Axioms

### Fix: manages => supervises

In the generated ontology the object property `manages` has range:

ObjectPropertyRange(<manages> ObjectUnionOf(<Department> <Project> <Team>))

consistent with the generated object propery assertion:

ObjectPropertyAssertion(<manages> <MaryJohnson> <ITDepartment>)

while the object property `supervises` has range:

ObjectPropertyRange(<supervises> <Employee>)

consistent for instance with the generated object propery assertions:

ObjectPropertyAssertion(<supervises> <MaryJohnson> <AliceGreen>).

Those two object propeties `manages` for something and `supervises` for someone,
are clearly disjointed.

Claude code generated also the following axiom: 

SubObjectPropertyOf(<manages> <supervises>)

from which we can derive the conseguent axiom:

ObjectPropertyAssertion(<supervises> <MaryJohnson> <ITDepartment>)

implies that the individual `ITDepartment` is an Employee, and we don't want that.

### Fix: hasEmail, hasName, hasEmployeeID data properties relationships

In the generated ontology we had the axioms:

SubDataPropertyOf(<hasEmail> <hasName>)

EquivalentDataProperties(<hasEmployeeID> <hasName>)

According to them, an email associated to a person can be seen also as name of the same person,
or all employee IDs are also names and vice versa all the names are also employe IDs.
Putting the two axiom together we can say that all emails are also employees IDs.
We prefer to keep those concept disjoint in our ontology.

### ObjectProperties Reasoner Limitations

It looks that the reasoner does not support:

1. Disjoint
2. InverseFunctional
3. Asymmetric
4. Irreflexive
5. Cardinality

**object property** axioms.

This is an error produced by the reasoner:

```bash
ERROR  14:29:09  An error occurred during reasoning: Non-simple property '<http://www.semanticweb.org/fax/ontologies/2026/corporate#supervises>' or its inverse appears in disjoint properties axiom..
java.lang.IllegalArgumentException: Non-simple property '<http://www.semanticweb.org/fax/ontologies/2026/corporate#supervises>' or its inverse appears in disjoint properties axiom.
``` 

So, I manually removed those axioms:

* DisjointObjectProperties(<colleagueOf> <supervises>)
* DisjointObjectProperties(<managedBy> <manages>)
* DisjointObjectProperties(<mentors> <supervises>)
* DisjointObjectProperties(<supervisedBy> <supervises>)
* InverseFunctionalObjectProperty(<:supervises>)
* AsymmetricObjectProperty(<supervises>)
* IrreflexiveObjectProperty(<supervises>)
* SubClassOf(<Employee> ObjectExactCardinality(1 <supervisedBy> <Manager>))
