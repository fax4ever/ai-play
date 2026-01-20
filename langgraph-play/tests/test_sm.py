from langgraph_play.lg_sm import StateMachine

def test_state_machine():
    sm = StateMachine()
    
    # Config loaded
    assert sm.config["settings"]["initial_state"] == "greet"
    print("✓ Config loaded")
    
    # Initial state creation
    state = sm.create_initial_state()
    assert state["current_state"] == "greet"
    assert state["messages"] == []
    assert state["user_name"] is None
    print("✓ Initial state created")
    
    # State type checks
    assert sm.is_waiting_state("greet") == True
    assert sm.is_waiting_state("process_input") == False
    assert sm.is_terminal_state("end") == True
    print("✓ State type checks")
    
    # Text formatting
    assert sm._format_text("Hello {user_name}", {"user_name": "Alice"}) == "Hello Alice"
    assert sm._format_text("Hi {{name}}", {}) == "Hi {name}"  # escaped braces
    print("✓ Text formatting")
    
    print("\n✅ All tests passed!")
