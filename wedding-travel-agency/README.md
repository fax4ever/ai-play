# Wedding Travel Agency

PDDL planning project, made to be run with the ENHSP planner.

Clone the repository: 

``` bash
git clone https://gitlab.com/enricos83/ENHSP-Public.git
```

``` bash
cd ENHSP-Public/
```

Compile it

``` bash
./compile
```

Run the planner

``` shell
java -jar enhsp-dist/enhsp.jar -o /home/fax/code/ai-play/wedding-travel-agency/wta-domain.pddl -f /home/fax/code/ai-play/wedding-travel-agency/wta-problem-3.pddl -planner opt-hmax
```

In this project, we tested:

3 different problems:

1. wta-problem-1.pddl
2. wta-problem-2.pddl
3. wta-problem-3.pddl

with 3 different planner euristics:

1. opt-blind
2. sat-hadd
3. opt-hmax

