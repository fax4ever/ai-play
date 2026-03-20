# DL Queries

## Class Hierarchies

1. Person > Employee > Manager > Executive > CEO

+ Equivalent classes
+ Subclasses
+ Instances

## Derived Class

2. SeniorManager
*equivalentTo*
```sql
Manager and (hasYearsOfExperience some xsd:integer[>= 10])
```

3. ExperiencedEmployee
*equivalentTo*
```sql
Employee and (hasYearsOfExperience some xsd:integer[>= 5])
```

4. ProjectLead
*equivalentTo*
```sql
Employee and (manages only Project)
```

5. Worker
*equivalentTo*
```sql
Contractor or Employee
```

6. CompanySize
*equivalentTo*
```
{LargeCompany , MediumCompany , SmallCompany}
```

7. TeamMember
*equivalentTo*
```
Person and (worksOn some Project)
```

8. SmallTeam
*subclassOf*
```
hasMember exactly 3 Employee
```

9. LargeTeam
*subclassOf*
```
hasMember min 10 Person
```

## And Queries

```sql
Manager and (worksOn some Project)
```

```sql
Employee and (worksIn value ITDepartment) and (hasYearsOfExperience some xsd:integer[>= 5])
```

## Or Queries

```sql
Manager or Intern
```

```sql
worksIn value ITDepartment or worksIn value HRDepartment_Ind
```

## Not Queries

```sql
Employee and (not Manager)
```

```sql
Person and (not (supervises some Person))
```

### Existential Queries

```sql
supervises some Employee
```

```sql
supervisedBy some Manager
```

```sql
worksIn some Department
```

```sql
worksOn some Project
```

```sql
colleagueOf some Person
```

```sql
manages some owl:Thing
```

```sql
Employee and (hasOffice some Office)
```

### Universal Queries

```sql
manages only Project
```

```sql
Employee and (worksOn only ActiveProject)
```

```sql
Manager and (supervises only Employee)
```

### Values Queries

```sql
worksIn value ITDepartment
```

```sql
reportsTo value JohnSmith
```

```sql
inverse(hasMember) value DevTeam
```

### Inverse

```sql
inverse(supervises) value JohnSmith
```

```sql
inverse(colleagueOf) value BobWilson
```

### Self Restrictions

```sql
knows Self
```

```sql
supervises Self
```

```sql
colleagueOf Self
```

### Cardinality Restrictions

```sql
Manager and (supervises min 2 owl:Thing)
```

```sql
Team and (hasMember exactly 3 Employee)
```

```sql
ExperiencedEmployee and (supervises min 2 Person)
```

### Data Propery Restrictions

```sql
Employee and (hasSalary some xsd:integer[> 100000])
```

```sql
Person and (hasAge some xsd:integer[< 30])
```

```sql
Employee and (hasYearsOfExperience some xsd:integer[>= 10])
```

```sql
Person and (hasAge some xsd:integer[>= 25, <= 40])
```

```sql
Employee and (hasSalary some xsd:integer[> 80000]) and (hasYearsOfExperience some xsd:integer[> 5])
```

```sql
(Employee and (hasAge some xsd:integer[< 30])) and (worksIn value ITDepartment) and (worksOn some Project)
```
