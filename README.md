This repo contains mainly codes for homework from PhD course at INSEAD

## Course Recap
Session 1: Event study.
- Many instructions can be found on WRDs about [Even Studies](https://wrds-web.wharton.upenn.edu/wrds//ds/wrdseventstudy/ciq/index.cfm). 
Session 2: Capital structure and payout policies.
- How firm characteristics can affect the capital structure of the firm. 
Session 3: Investment
- Whether capital supply has an effect on investment and other decisions, focus on learning channel. 
 
## Project Log
### Preliminary findings
#### Data

- WRDs is able to produce a [graphic summary](https://wrds-www.wharton.upenn.edu/query-manager/query-document/901368/) of the event you are interested in. In particular, it plots CAR within the event window, together with CI. It also generates breakdown of everyday CAR table as well as event date list. Now I am trying to download them.
- To refer to Adrian's notes, we can compute J_1 to test if the average CAR is significant.

- I stratigize this project around a sample code for event study on [WRDS](https://wrds-www.wharton.upenn.edu/pages/support/applications/python-replications/programming-python-wrds-event-study/), which requires an input dictionary of event time and company permno, this is intuitive because when it comes to calculating CAR, any additional information like the event type does not really matter. I go on to [Key Development](https://wrds-web.wharton.upenn.edu/wrds//ds/comp/ciq/keydev/index.cfm) to query the needed data. For lawsuit the event key id is **25**.

- [Lawsuits](https://wrds-www.wharton.upenn.edu/query-manager/query/3871927/) definition in WRDs: this pertains to actions or suits brought before a court against the company, as to recover a right or redress a grievance.