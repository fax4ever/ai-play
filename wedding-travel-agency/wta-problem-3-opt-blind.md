# 3 opt-blind

> java -jar enhsp-dist/enhsp.jar -o /home/fax/code/ai-play/wedding-travel-agency/wta-domain.pddl -f /home/fax/code/ai-play/wedding-travel-agency/wta-problem-3.pddl -planner opt-blind

``` bash
-------------Time: 1424s ; Expanded Nodes: 17448122 (Avg-Speed 12252.0 n/s); Evaluated States: 24244450
Exception in thread "main" java.lang.OutOfMemoryError: Java heap space
	at com.carrotsearch.hppc.DoubleArrayList.clone(DoubleArrayList.java:435)
	at com.hstairs.ppmajal.PDDLProblem.PDDLState.<init>(PDDLState.java:51)
	at com.hstairs.ppmajal.PDDLProblem.PDDLState.clone(PDDLState.java:121)
	at com.hstairs.ppmajal.PDDLProblem.PDDLState.clone(PDDLState.java:39)
	at com.hstairs.ppmajal.PDDLProblem.PDDLProblem$stateIterator.hasNext(PDDLProblem.java:1519)
	at com.hstairs.ppmajal.search.WAStar.search(WAStar.java:129)
	at com.hstairs.ppmajal.PDDLProblem.PDDLPlanner.plan(PDDLPlanner.java:85)
	at planners.ENHSP.search(ENHSP.java:545)
	at planners.ENHSP.planning(ENHSP.java:209)
	at main.main(main.java:31)
```