# A structured scene assessment produced by the VLM (Module 4).
builtin_interfaces/Time stamp
string   caption          # one-sentence description of the scene
string[] labels           # observed object/condition labels
string   risk_level       # "none" | "low" | "medium" | "high"
string   rationale        # why this risk level
string   recommendation   # suggested action (e.g. "reroute", "notify operator")
float32  confidence       # 0..1
string   backend          # which VLM backend produced this
