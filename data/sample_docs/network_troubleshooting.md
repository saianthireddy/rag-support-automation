# Network Troubleshooting Guide

## Diagnosing connectivity loss

If the unit drops off the network, check the link LED on the rear port first.
A steady amber light means the interface negotiated a link but failed DHCP.
Power-cycling the switch port resolves most of these cases; a full restart of
the appliance is rarely required and takes it offline for roughly two minutes.

## Interpreting network error codes

Error code N-22 indicates a DNS resolution failure and is almost always a
customer-side configuration problem. Error code N-31 indicates packet loss
above five percent on the uplink and should be raised with the network team
rather than handled in the support queue.

## Static addressing

Static IP assignment lives in the admin console under Settings > Network >
Interfaces. Record the previous address before changing it, since a bad static
assignment locks the console out and requires physical access to recover.
