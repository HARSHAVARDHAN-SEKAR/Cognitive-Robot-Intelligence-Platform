# A single task in a mission plan produced by the LLM brain.
string action          # "navigate_to" | "wait" | "report" | "search_for"
string target_label    # semantic label this task refers to (may be empty)
geometry_msgs/Point target_point   # resolved map-frame goal (for navigate_to)
float32 duration       # seconds (for wait)
string note            # human-readable description / rationale
