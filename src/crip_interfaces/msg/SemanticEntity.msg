# A single semantic entity stored in the world model.
string   label          # e.g. "person", "charging_station", "box_obstacle"
string   category       # e.g. "static", "dynamic", "hazard", "landmark"
geometry_msgs/Point position   # map-frame position (metres)
float32  confidence     # 0..1
int32    observation_count   # how many times this entity has been seen
builtin_interfaces/Time last_seen
int32    id             # database id (0 if not yet persisted)
