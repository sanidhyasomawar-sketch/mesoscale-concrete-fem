"""Comprehensive end-to-end test of Streamlit app navigation and page rendering."""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from streamlit.testing.v1 import AppTest

def find_button_by_label_part(at, part):
    for b in at.button:
        if part in b.label:
            return b
    return None

def find_button_by_key(at, key):
    for b in at.button:
        if b.key == key:
            return b
    return None

print("=== STARTING COMPREHENSIVE STREAMLIT APP NAVIGATION TEST ===")

# 1. Initialize app
at = AppTest.from_file("app.py", default_timeout=30)
at.run()
assert len(at.exception) == 0, f"App failed to start: {[e.message for e in at.exception]}"
assert at.sidebar.radio[0].value == "Home", f"Expected Home, got {at.sidebar.radio[0].value}"
print("✓ Test 1: App starts at Home without exception")

# 2. Verify landing page elements
hero_btn = find_button_by_key(at, "launch_app_hero_btn")
bot_btn = find_button_by_key(at, "launch_app_bot_btn")
assert hero_btn is not None, "Hero Launch button not found"
assert bot_btn is not None, "Bottom Launch button not found"
print("✓ Test 2: Landing page buttons rendered correctly")

# 3. Test Hero 'Launch Application →' button
hero_btn.click()
at.run()
assert len(at.exception) == 0, f"Exception on Launch click: {[e.message for e in at.exception]}"
assert at.sidebar.radio[0].value == "Phase 1: Mix Design", f"Expected Phase 1, got {at.sidebar.radio[0].value}"
print("✓ Test 3: Hero 'Launch Application →' navigated to Phase 1: Mix Design")

# 4. Verify Phase 1 page content and return button
return_btn = find_button_by_key(at, "sidebar_return_home_btn")
assert return_btn is not None, "Sidebar 'Return to Landing Page' button not found"
print("✓ Test 4: Phase 1 rendered with 'Return to Landing Page' button")

# 5. Test returning to Home
return_btn.click()
at.run()
assert len(at.exception) == 0, f"Exception on Return Home: {[e.message for e in at.exception]}"
assert at.sidebar.radio[0].value == "Home", f"Expected Home, got {at.sidebar.radio[0].value}"
print("✓ Test 5: Returned to Home successfully")

# 6. Test bottom 'Proceed to Phase 1: Mix Design →' button
bot_btn = find_button_by_key(at, "launch_app_bot_btn")
assert bot_btn is not None, "Bottom Launch button not found on Home"
bot_btn.click()
at.run()
assert len(at.exception) == 0, f"Exception on Bottom Launch click: {[e.message for e in at.exception]}"
assert at.sidebar.radio[0].value == "Phase 1: Mix Design", f"Expected Phase 1, got {at.sidebar.radio[0].value}"
print("✓ Test 6: Bottom 'Proceed to Phase 1' navigated to Phase 1: Mix Design")

# 7. Test direct radio navigation to Phase 2
at.sidebar.radio[0].set_value("Phase 2: Geometry Generation")
at.run()
assert len(at.exception) == 0, f"Exception on Phase 2: {[e.message for e in at.exception]}"
assert at.sidebar.radio[0].value == "Phase 2: Geometry Generation"
print("✓ Test 7: Direct radio navigation to Phase 2: Geometry Generation")

# 8. Test direct radio navigation to Phase 3
at.sidebar.radio[0].set_value("Phase 3: Conforming Mesh")
at.run()
assert len(at.exception) == 0, f"Exception on Phase 3: {[e.message for e in at.exception]}"
assert at.sidebar.radio[0].value == "Phase 3: Conforming Mesh"
print("✓ Test 8: Direct radio navigation to Phase 3: Conforming Mesh (empty state clean)")

# 9. Test direct radio navigation to Phase 4
at.sidebar.radio[0].set_value("Phase 4: Linear FEM")
at.run()
assert len(at.exception) == 0, f"Exception on Phase 4: {[e.message for e in at.exception]}"
assert at.sidebar.radio[0].value == "Phase 4: Linear FEM"
print("✓ Test 9: Direct radio navigation to Phase 4: Linear FEM (empty state clean)")

# 10. Test direct radio navigation to Phase 5
at.sidebar.radio[0].set_value("Phase 5: Cohesive ITZ")
at.run()
assert len(at.exception) == 0, f"Exception on Phase 5: {[e.message for e in at.exception]}"
assert at.sidebar.radio[0].value == "Phase 5: Cohesive ITZ"
print("✓ Test 10: Direct radio navigation to Phase 5: Cohesive ITZ (empty state clean)")

# 11. Return to Home via sidebar button from Phase 5
return_btn = find_button_by_key(at, "sidebar_return_home_btn")
assert return_btn is not None, "Return button missing on Phase 5"
return_btn.click()
at.run()
assert len(at.exception) == 0, f"Exception returning Home from Phase 5: {[e.message for e in at.exception]}"
assert at.sidebar.radio[0].value == "Home"
print("✓ Test 11: Return to Home from Phase 5 works")

# 12. Repeated Launch Application click
hero_btn = find_button_by_key(at, "launch_app_hero_btn")
hero_btn.click()
at.run()
assert len(at.exception) == 0, f"Exception on repeated Launch click: {[e.message for e in at.exception]}"
assert at.sidebar.radio[0].value == "Phase 1: Mix Design"
print("✓ Test 12: Repeated Launch Application navigation confirmed working without exception")

print("\n=== ALL 12 NAVIGATION & UI TESTS PASSED SUCCESSFULLY! ===")
