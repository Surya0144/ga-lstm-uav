# uav_network.tcl
# NS2 FANET Simulation Configuration (sources 130-153)

set ns [new Simulator]

# Setup trace files for network metrics (Implied by ns2_python_bridge.py source 167)
set tracefile [open network.tr w]
$ns trace-all $tracefile
set namfile [open network.nam w]
$ns namtrace-all $namfile

# --- Configuration from report ---
set num_uavs 5
set simulation_time 100.0

# Configure wireless network with AODV routing
$ns node-config -adhocRouting AODV \
                 -llType LL \
                 -macType Mac/802_11 \
                 -ifqType Queue/DropTail/PriQueue \
                 -antType Antenna/OmniAntenna \
                 -propType Propagation/TwoRayGround \
                 -phyType Phy/WirelessPhy

# Create UAV nodes with 3D mobility
for {set i 0} {$i < $num_uavs} {incr i} {
    set uav($i) [$ns node]
    # Set initial random positions (sources 146-152)
    $uav($i) set X_ [expr rand() * 1000]
    $uav($i) set Y_ [expr rand() * 1000]
    $uav($i) set Z_ [expr rand() * 400 + 100]
}

# --- End of configuration ---

# This procedure is called by NS2 to stop the simulation
proc finish {} {
    global ns tracefile namfile
    $ns flush-trace
    close $tracefile
    close $namfile
    # Don't exit; let the Python bridge control the lifecycle
    puts "NS2 simulation finished."
}

# Schedule the 'finish' event
$ns at $simulation_time "finish"

puts "NS2 simulation script (uav_network.tcl) loaded."
# Note: The Python bridge will start and control this script.
# We add '$ns run' here so the simulation starts when called.
$ns run