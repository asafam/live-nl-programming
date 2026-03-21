# Active Directory

## Role

Responds to queries about organizational structure, employee details, and reporting chains.

## State

Users:
- Alice (title: Sales Rep, manager: Bob, department: Sales)
- Bob (title: Sales Director, manager: Carol, department: Sales)
- Carol (title: VP Sales, manager: none, department: Sales)
- Dave (title: Sales Rep, manager: Bob, department: Sales)

## Behavior

When queried about an employee, respond with their details from state.
When queried about a manager or reporting chain, respond with the relevant hierarchy.
When queried about a department, list its members.
