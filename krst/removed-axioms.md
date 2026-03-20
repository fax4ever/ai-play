# Removed Axioms

## DisjointObjectProperties

It looks that the reasoner does not support disjoint object property axioms:

* DisjointObjectProperties(<#colleagueOf> <#supervises>)
* DisjointObjectProperties(<#managedBy> <#manages>)
* DisjointObjectProperties(<#mentors> <#supervises>)
* DisjointObjectProperties(<#supervisedBy> <#supervises>)

```bash
ERROR  14:29:09  An error occurred during reasoning: Non-simple property '<http://www.semanticweb.org/fax/ontologies/2026/corporate#supervises>' or its inverse appears in disjoint properties axiom..
java.lang.IllegalArgumentException: Non-simple property '<http://www.semanticweb.org/fax/ontologies/2026/corporate#supervises>' or its inverse appears in disjoint properties axiom.
``` 

## InverseFunctionalObjectProperty

It looks that the reasoner does not support inverse functional object property axioms:

* InverseFunctionalObjectProperty(<#supervises>)

## AsymmetricObjectProperty

It looks that the reasoner does not support asymmetric object property axioms:

* AsymmetricObjectProperty(<#supervises>)

## IrreflexiveObjectProperty

It looks that the reasoner does not support irreflexive object property axioms:

* IrreflexiveObjectProperty(<#supervises>)

## ObjectExactCardinality

It looks that the reasoner does not support object exact cardinality object property axioms:

* SubClassOf(<#Employee> ObjectExactCardinality(1 <#supervisedBy> <#Manager>))
* SubClassOf(ObjectHasValue(<#worksIn> <#ITDepartment>) <#Employee>)
