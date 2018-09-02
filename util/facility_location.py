from scipy.spatial.distance import pdist, squareform

# Solves a facility location problem using the algorithm from Jain et al. "Greedy Facility Location Algorithms Analyzed
# using Dual Fitting with Factor-Revealing LP".  A facility location algorithm takes as input a graph G=(V,E), where
# each edge is a cost for connecting two vertices (or "cities").  The goal is to choose a subset of vertices (or
# "facilities"), such that each city is connected to exactly one facility, while minimizing the total cost.  The
# cost is the sum of selected edge connections, plus predefined cost for opening up each facility.  The algorithm
# of Jain et al. is an O(E log E) algorithm with an approximation guarantee of within 1.61 times the optimal
# solution
#
# Let cityFacilityCosts be a list of 3-tuples, where each element (cost,facility,city) is the cost
# of connecting a city to a facility and openFacilityCost is the cost of opening a new facility.
# Returns a dict where retval[facility][city] is present for connected facility-city pairs, with value equal
# to the connection cost.
class FacilityLocation():
    def __init__(self, points=None, cityFacilityCosts=None, cityDisallowedCityNeighbors=None):
        """
        points : np.array of points
        cityDisallowedCityNeighbors : for city j and city k, cityDisallowedCityNeighbors[j][k] = -1 if
            the cities cannot be neighbors
        """

        self.points = points
        self.cityFacilityCosts = cityFacilityCosts
        self.costs_s = None
        self.allFacilities = {}
        self.allCities = {}
        self.city_disallowed_city_neighbors = cityDisallowedCityNeighbors
        if self.points is not None:
            for i in range(0,len(self.points)):
                self.allFacilities[i] = self.allCities[i] = 1
        else:
            for c in cityFacilityCosts:
                if not c[1] in self.allFacilities: self.allFacilities[c[1]] = 1
                if not c[2] in self.allCities: self.allCities[c[2]] = 1

    def solve(self, openFacilityCost=None, openFacilityCosts=None, debug=0):
        """
        openFacilityCost: cost of opening a facility
        openFacilityCosts: dict mapping facility to the cost of opening it
        """

        if openFacilityCosts is None:
            # Initialize all facility costs to `openFacilityCost`
            openFacilityCosts = {}
            for f in self.allFacilities:
                openFacilityCosts[f] = openFacilityCost
        self.openFacilityCosts = openFacilityCosts

        # If the user didn't give us `cityFacilityCosts` then compute it from `points`.
        if self.cityFacilityCosts is None:
            self.compute_costs()

        # A priority queue keyed by facility, with value set to the value of alpha when that facility should be opened
        self.fac_open_candidates = priority_dict()

        # A dict keyed by facility, where each value is a 4-tuple: (S1,S2,n1,n2), where S1 is the sum of offers from
        # unconnected cities--excluding alpha terms--to this facility, n1 is the number of offers from unconnected
        # cities, S2 is the sum of switching offers from connected cities, and n2 is the number of offers from
        # connected cities.  Keeping this structure allows us to compute at any given time the value of alpha
        # at which a facility should be opened
        self.facility_offers = {}

        self.facility_cities = {} # Return value, a dict where the key is a facility and each value is a list of connected cities
        self.city_facilities = {} # A dict keyed by city, with the value set to the facility that city is connected to
        self.fac_city_offers = {} # For each facility f and city c, fac_city_offers[f][c] is an offer to connect c to f
        self.city_fac_offers = {} # For each facility f and city c, city_fac_offers[c][f] is an offer to connect c to f
        self.facility_disallowed_cities = {} # For each facility f and city c, facility_disallowed_cities[f][c] = [n, v] where n is the number of cities disallowing c, and v is the current offer of c to f
        self.total_cost = 0

        # Now sort the object-object pairs by increasing matching cost, and iterate through each one.  This enables us
        # to monotonically increase alpha, while keeping track of exactly the connection offers that are non-zero
        if self.costs_s is None:
            self.costs_s = sorted(self.cityFacilityCosts, key=lambda t: t[0])

        # If a facility is put on the fac_open_candidates with only switch candidates, then make the alpha value
        # for opening the facility equal to `(max cost of connecting a facility and city) + 1`
        self.big = 0
        for k in range(0, len(self.costs_s)):
            if len(self.costs_s[k]) > 0:
                self.big = max(self.big, self.costs_s[k][0] + 1)

        # Go through each (cost, facility, city) item
        for k in range(0, len(self.costs_s)):

            cost = self.costs_s[k][0]
            facility = self.costs_s[k][1]
            city = self.costs_s[k][2]

            # Has this city already been connected to a facility?
            if city in self.city_facilities:
                continue

            # The next value for alpha will be equal to cost.
            # However, for some unopened facility, the total offers it receives reaches the cost of opening a facility before
            # alpha=cost.  Open that facility
            while len(self.fac_open_candidates) > 0:
                # get the value of alpha that will cause this facility to open
                #print str(self.fac_open_candidates)
                alpha = self.fac_open_candidates[self.fac_open_candidates.smallest()]

                # If alpha is greater than the current cost, then move on
                if alpha > cost:
                    break

                # If alpha is less than or equal to the cost, then open the facility
                self.open_facility(self.fac_open_candidates.pop_smallest(), alpha, debug)

            # Have this city make an offer to the facility
            if debug > 1:
                self.debug_offers()
            self.offer(city, facility, cost, debug)
            if debug > 1:
                self.debug_offers()

        # If it happens that some cities are still not connected, then open up more facilities.
        while len(self.city_facilities) < len(self.allCities):

            ###############
            # DEBUGGING
            # self.debug_offers()
            # r = raw_input("enter...")
            # END DEBUGGING
            ##############

            if debug > 0:
                print "Opening facility because some cities were unconnected"

            facility = self.fac_open_candidates.smallest()
            alpha = self.fac_open_candidates[facility]

            min_to_open = self.facility_offers[facility][2]
            prev_open_cities = len(self.city_facilities)

            self.open_facility(self.fac_open_candidates.pop_smallest(), alpha, debug)

            assert len(self.city_facilities) >= prev_open_cities + min_to_open

        return (self.facility_cities, self.total_cost)

    def offer(self, city, facility, cost, debug):
        """ Have the `city` offer some contribution to the `facility`
        """

        # Is this city already connected to another facility? Then we don't need to make a
        # more expensive offer (equal to `cost`) to another facility.
        if city in self.city_facilities:
            return

        # Can this city and facility be connected?
        if self.can_facility_and_city_be_connected(facility, city):

            # Offer city as a candidate connection to facility
            if debug > 0:
                print "Offer city " + str(city) + " to facility " + str(facility) + " at cost " + str(cost)

            # Initialize the facility offer data structure that allows us to compute `alpha` (the price at which to open this facility)
            if facility not in self.facility_offers:
                self.facility_offers[facility] = [0,0,0,0]

            # Add in the offer from this city
            # `city` is unconnected, so we update t[0] and t[2]
            # (t[1] and t[3] are for switching offers...)
            t = self.facility_offers[facility]
            t[0] -= cost
            t[2] += 1

            # Initialize the city offer data structure to track which facilities `city` has offers to
            if city not in self.city_fac_offers:
                self.city_fac_offers[city]={}
            self.city_fac_offers[city][facility] = (cost,None,None) # This signifies the offer as an "unconnected" offer as opposed to "switching"

            # Initialize the facility offer data structure to track which cities are making offers to `facility`
            if facility not in self.fac_city_offers:
                self.fac_city_offers[facility]={}
            self.fac_city_offers[facility][city] = cost


            # If this facility is already opened, then just connect the city to it.
            if facility in self.facility_cities:

                # Connect city to existing facility. This is the closest open facility
                if debug > 0:
                    print "Connect city " + str(city) + " to existing facility " + str(facility) + " at cost " + str(cost)
                # NOTE: connect() calls `update_disallowed_neighbors_addition`
                self.connect(facility, city, debug)


            # This facility is unopened, we need to update the `alpha` at which to open it.
            else:

                # Update the value of `alpha` that will cause `facility` to open.
                assert t[2] > 0, "We just made an offer, this value should be greater than 0"
                # open_alpha = (cost of opening) - (SUM(offers from unconnected cities)) - (SUM(switching offers from connected cities)) / (SUM(number of unconnected cities with offers))
                # The division is for computing the "cost effecitiveness" of opening the facility.
                self.fac_open_candidates[facility] = (self.openFacilityCosts[facility]-t[0]-t[1]) / t[2]

                # Update which neighbors cannot be added to this facility
                self.update_disallowed_neighbors_addition(facility, city, debug)



        # Has a previous city disallowed this connection?
        else:
            if facility in self.facility_disallowed_cities and city in self.facility_disallowed_cities[facility]:
                if self.facility_disallowed_cities[facility][city][0] > 0:
                    if debug > 0:
                        print "disallowed offer city " + str(city) + " to facility " + str(facility) + " at cost " + str(cost)
                    # Store the cost of not being able to connect this city to the factory because some
                    # other disallowd neighbor is already connected. If that neighbor removes their offer
                    # to the facility, then we'll use this cost to make an offer
                    self.facility_disallowed_cities[facility][city][1] = cost


    def open_facility(self, facility, alpha, debug):
        """Open a new facility and connect all candidate cities.
        """
        assert facility not in self.facility_cities

        self.facility_cities[facility] = {}
        if facility not in self.facility_disallowed_cities:
            self.facility_disallowed_cities[facility] = {}

        opening_cost = self.openFacilityCosts[facility]
        self.total_cost += opening_cost

        if debug > 0:
            print "Open facility " + str(facility) + " (open cost = " + str(opening_cost) + ") at alpha " + str(alpha) + ":"

        # Connect each offering city c to facility
        for c in self.fac_city_offers[facility]:
            # c might be an unconnected city or it could be a connected city that is switching.
            self.connect(facility, c, debug)

        # Clear out all offers for this facility, since they were just connected
        # BUG: is this correct?
        del self.fac_city_offers[facility]

        #self.sanity_check_neighbors()
        #self.sanity_check_offers()

    def connect(self, facility, c, debug):
        """Connect `facility` to city `c`.
        Assumes a connection between them has already been offered.
        Assumes the facility is open.
        """

        # Get the offer that the city made to the facility.
        c_ip_j = self.fac_city_offers[facility][c]
        if debug > 0:
            print "  connect city " + str(c) + " to facility " + str(facility) + " at cost " + str(c_ip_j)

        # This city should not be in the disallowed cities for this facility
        if c in self.facility_disallowed_cities[facility] and self.facility_disallowed_cities[facility][c][0] > 0:
            raise Exception("A city is trying to connect to dissallowed facility.")

        # Delete this facility from the city's offers list
        del self.city_fac_offers[c][facility]

        # Connect the city to the facility, storing the contribution it made to the facility
        self.facility_cities[facility][c] = c_ip_j
        self.total_cost += c_ip_j

        # Keep a list of all the facilities that need to be updated for neighbor removal
        update_facilities_for_disallowed_neighbor_removal = []

        if c in self.city_facilities:
            # If c is already connected to a facility f, we must switch it
            f = self.city_facilities[c]
            self.total_cost -= self.facility_cities[f][c]
            if debug > 0:
                print "    requires switch from facility " + str(f) + " at cost " + str(self.facility_cities[f][c])
            del self.facility_cities[f][c]
            del self.city_facilities[c]

            # It will never be advantageous to switch back to `f`
            # So we do not need to update `city_fac_offers` (and `f` should not be in `self.city_fac_offers[c]`)
            assert f not in self.city_fac_offers[c], "`c` should not have an offer out to an opened facility"

            # We do need to update the disallowed city list for `f`
            update_facilities_for_disallowed_neighbor_removal.append(f)


        self.city_facilities[c] = facility
        self.update_disallowed_neighbors_addition(facility, c, debug)


        # The following loop is used to update `c`'s offer to all the other facilities.
        # For each facility, we first subtract `c`'s previous offer and then add back in the new offer.

        # city_fac_offers[c][f] = [cost, None, None] by default
        # city_fac_offers[c][f] = [cost, f, savings]
        bad = [] # collect the facilities that this city will never connect to (now that it is connected to `facility`)
        for f in self.city_fac_offers[c]:

            assert f != facility, "This should never be the case since we just deleted `facility` from `self.city_fac_offers[c]`"

            # Update the offers of city c to switch to other unopened facilities
            c_i_j = self.fac_city_offers[f][c] # get the contribution made by c to f
            t = self.facility_offers[f] # get the total contributions for f

            # Prep the `t` variable for updating by removing the contribution of `c`
            # The subsequent code block adds the updated offer back in
            if self.city_fac_offers[c][f][1] is not None:
                # NOTE: this is the swith case
                # We've already been in this loop and have updated the offer to a switch offer
                # So we need to remove the contribution to t[1] and decrement t[3]
                t[1] += self.city_fac_offers[c][f][2]  # remove c from sum offers to connect unconnected cities to f
                t[3] -= 1             # number of offers to connect unconnected cities to f
                if t[3] < 0 and debug > 1: # Should never decrement below 0
                    raise Exception("huh?")
            else:
                # NOTE: this is the unconnected case
                # This city made an offer to the facility when it was still unconnected.
                # So to update t, we need to remove the contribution to t[0], and decrement t[1]
                # In the next code block we'll be changing the offer to a switch offer
                t[0] += c_i_j         # remove c from sum switch offers
                t[2] -= 1             # number of switch
                if t[2] < 0 and debug > 1: # Should never decrement below 0
                    raise Exception("huh?")

            # The switch offer of `c` to `f` is now the savings that would occur by switching from `facility` to `f`
            switch_offer = c_ip_j - c_i_j
            make_switch_offer = True

            # If `f` has already been opened, but `c` was connected to `facility` instead, then it will never be better to connect to `f`
            if f in self.facility_cities:
                make_switch_offer = False

            # If the `switching_offer` is negative, then it will never be better to connect `c` to `f`
            if switch_offer < 0:
                make_switch_offer = False

            # Is `c` in the disallowed list for `f`? If so, don't make a switch offer.
            if f in self.facility_disallowed_cities and c in self.facility_disallowed_cities[f]:
                if self.facility_disallowed_cities[f][c][0] > 0: # some other offer is preventing `c`
                    make_switch_offer = False

            if make_switch_offer:
                if debug > 0:
                    print "    offer switch from facility " + str(facility) + "("+str(c_ip_j)+ ") to " + str(f) + "(" + str(c_i_j)+") for city " + str(c)

                # Add the offer to switch to the `facility_offers[f]` data structure
                t[1] += switch_offer  # sum offers to switch connections of already connected cities to f
                t[3] += 1             # number of offers to switch connections of already connected cities to f

                # Update `city_fac_offers` to reflect that we are now making a switching offer (as opposed to an un-connected offer)
                self.city_fac_offers[c][f] = (c_i_j,f,c_i_j-c_ip_j)

            # `c` will never be matched to `f`
            else:
                # Add `f` to the list of factories `c` will never connect to.
                bad.append(f)
                update_facilities_for_disallowed_neighbor_removal.append(f)

            # Regardless of whether we made a switch offer, make sure to update the alpha value that will open this facility.
            # If there are no unconnected cities making an offer, then set alpha to a big number
            if t[2] > 0:
                self.fac_open_candidates[f] = (self.openFacilityCosts[f]-t[0]-t[1]) / t[2]
            else:
                self.fac_open_candidates[f] = np.inf #self.big

        # Remove `c` from all factories `f` that it will never connect to
        for f in bad:
            del self.fac_city_offers[f][c]
            del self.city_fac_offers[c][f]

        # Update which neighbors can now connect / make offers to f
        for f in update_facilities_for_disallowed_neighbor_removal:
            self.update_disallowed_neighbors_removal(f, c, debug)


    def can_facility_and_city_be_connected(self, facility, city):

        can_be_connected = True
        if facility in self.facility_disallowed_cities:
            if city in self.facility_disallowed_cities[facility]:
                # Is there some other city preventing `city` from being connected to `facility`?
                if self.facility_disallowed_cities[facility][city][0] > 0:
                   can_be_connected = False

        return can_be_connected


    def update_disallowed_neighbors_addition(self, facility, added_city, debug):
        """ Update which cities cannot make offers to `facility` now that `added_city` has made an offer / is connected.
        """

        if (self.city_disallowed_city_neighbors is not None) and (added_city in self.city_disallowed_city_neighbors):

            # Go through the disallowed neighbors
            for c2 in self.city_disallowed_city_neighbors[added_city]:

                # The value for `self.city_disallowed_city_neighbors[c1][c2]` is traditionally `-1`
                dummy_facility = self.city_disallowed_city_neighbors[added_city][c2]

                # NOTE: could make this more generic in the future
                assert type(dummy_facility) is int, "Invalid type for `city_disallowed_city_neighbors[c1][c2] value"

                # A bit of a hack: it is okay for these cities to share the dummy facility....
                if facility != dummy_facility:

                    # Connecting `added_city` to `facility` requires that `c2` can not be connected to `facility`
                    if facility not in self.facility_disallowed_cities:
                        self.facility_disallowed_cities[facility] = {}
                    if c2 not in self.facility_disallowed_cities[facility]:
                        self.facility_disallowed_cities[facility][c2] = [0,0]
                    self.facility_disallowed_cities[facility][c2][0] += 1
                    if debug > 0:
                        print "  offer " + str(added_city) + " disallow " + str(c2) + " from " + str(facility) + " count=" + str(self.facility_disallowed_cities[facility][c2][0])


    def update_disallowed_neighbors_removal(self, facility, removed_city, debug):
        """ Update which cities can make offers to `facility` now that `removed_city` is gone.
        """
        if (self.city_disallowed_city_neighbors is not None) and (removed_city in self.city_disallowed_city_neighbors):

            for c2 in self.city_disallowed_city_neighbors[removed_city]:

                if c2 in self.facility_disallowed_cities[facility]:

                    self.facility_disallowed_cities[facility][c2][0] -= 1

                    # Have all disallowed neighbors for `c2` been unconnected from `f`?
                    if self.facility_disallowed_cities[facility][c2][0] == 0:
                        # Does `c2` have an offer to `f`?
                        if self.facility_disallowed_cities[facility][c2][1] != 0:
                            # Is `c2` not already connected to another facility?
                            if c2 not in self.city_facilities:
                                if debug > 0:
                                    print "  undisallow " + str(c2) + " from " + str(facility) + " ( "+str(removed_city)+" was connected) cost=" + str(self.facility_disallowed_cities[facility][c2][1])
                                # Have `c2` make an offer to `f`
                                self.offer(c2, facility, self.facility_disallowed_cities[facility][c2][1], debug)

    def sanity_check_neighbors(self):
        # Sanity check the disallowed neighbors
        if self.city_disallowed_city_neighbors is not None:
            # Go through the current assignments
            for facility, cities in self.facility_cities.items():
                # Skip the dummy facility
                if facility == -1:
                    continue
                # Go through each city and ensure that none of the other connected cities are in it's disallowed list
                for c1 in cities:
                    if c1 in self.city_disallowed_city_neighbors:
                        for c2 in cities:
                            if c1 == c2:
                                continue
                            assert c2 not in self.city_disallowed_city_neighbors[c1], "Disallowed cities are connected %d & %d in fac %d." % (c1, c2, facility)

    def sanity_check_offers(self):

        for city, facility in self.city_facilities.items():

            c_ip_j = self.facility_cities[facility][city]

            for f in self.city_fac_offers[city]:

                assert f != facility, "This should never be the case since we just deleted `facility` from `self.city_fac_offers[c]`"

                # Update the offers of city c to switch to other unopened facilities
                c_i_j = self.fac_city_offers[f][city]

                assert c_ip_j >= c_i_j

    def compute_costs(self, type='euclidean'):
        """ Compute the pairwise distance between the points and use them as the cost
        between facilities and cities.
        """
        C = squareform(pdist(self.points, type))
        self.cityFacilityCosts = []
        for i in range(0,C.shape[0]):
            for j in range(0,C.shape[0]):
                self.cityFacilityCosts.append((C[i,j], i,j))
        self.costs_s = None

    def debug_offers(self):

        print "#############################"
        print "DEBUG OFFERS"
        print "%d facilities are open" % (len(self.facility_cities),)
        print "%d / %d cities are connected" % (len(self.city_facilities), len(self.allCities))

        print ""
        print "#############################"
        print "START --facility_cities "# + str(self.facility_cities)
        #print self.facility_cities
        fc = self.facility_cities.items()
        fc.sort(key=lambda x: x[0])
        for facility, cities in fc:
            cc = cities.items()
            cc.sort(key=lambda x: x[0])
            print "Facility %d is connect to %d cities" % (facility, len(cc))
            for city, cost in cc:
                print "\t%d at cost %0.3f" % (city, cost)
        print "END --facility_cities "
        print "#############################"
        print
        print "#############################"
        print "START --city_facilities "# + str(self.city_facilities)
        #print self.city_facilities
        cf = self.city_facilities.items()
        cf.sort(key=lambda x: x[0])
        for city, facility in cf:
            print "City %d connected to Facility %d" % (city, facility)
        print "END --facility_cities "
        print "#############################"
        print
        print "#############################"
        print "START --fac_open_candidates " #+ str(self.fac_open_candidates)
        #print self.fac_open_candidates
        fa = self.fac_open_candidates.items()
        fa.sort(key=lambda x: x[1])
        for facility, alpha in fa:
            unopened = self.facility_offers[facility][2]
            switches = self.facility_offers[facility][3]
            assert len(self.fac_city_offers[facility]) == (unopened + switches)
            print "Facility %d can open with alpha %0.3f (connecting %d cities (%d unopened and %d switches))" % (facility, alpha, len(self.fac_city_offers[facility]), unopened, switches)
        print "END --fac_open_candidates "
        print "#############################"
        print
        # print "#############################"
        # print "START --fac_city_offers " #+ str(self.fac_city_offers)
        # #print self.fac_city_offers
        # fc = self.fac_city_offers.items()
        # fc.sort(key=lambda x: x[0])
        # for facility, offers in fc:
        #     co = offers.items()
        #     co.sort(key=lambda x: x[0])
        #     print "Facility %d has offers from %d cities:" % (facility, len(co))
        #     for city, offer in co:
        #         print "\tCity %d with offer %0.3f" % (city, offer)
        # print "END --fac_city_offers "
        # print
        # print "#############################"
        # print "START --city_fac_offers " #+ str(self.city_fac_offers)
        # #print self.city_fac_offers
        # co = self.city_fac_offers.items()
        # co.sort(key=lambda x: x[0])
        # for city, offers in co:
        #     fo = offers.items()
        #     fo.sort(key=lambda x: x[0])
        #     print "City %d has offers to %d facilities:" % (city, len(fo))
        #     for facility, (offer, f, diff) in fo:
        #         print ("\tFacility %d with offer %0.3f" % (facility, offer)) + (" (diff is %0.3f to %d)" % (diff, f) if diff is not None else "")
        # print "END --city_fac_offers "
        # print "#############################"
        # print
        # print "#############################"
        # print "START --facility_offers " #+ str(self.facility_offers)
        # #print self.facility_offers
        # fo = self.facility_offers.items()
        # fo.sort(key=lambda x: x[0])
        # for facility, (s1, s2, n1, n2) in fo:
        #     print "Facility %d offer: [%0.3f, %0.3f, %d, %d]" % (facility, s1, s2, n1, n2)
        # print "END --facility_offers "
        # print "#############################"
        # print
        # print "#############################"
        # print "START --facility_disallowed_cities" #+ str(self.facility_disallowed_cities)
        # #print self.facility_disallowed_cities
        # fc = self.facility_disallowed_cities.items()
        # fc.sort(key=lambda x: x[0])
        # for facility, cities in fc:
        #     cc = cities.items()
        #     cc.sort(key=lambda x: x[0])
        #     print "Facility %d cannot be connected to %d cities:" % (facility, len(cc))
        #     for city, (num_against, cost) in cc:
        #         print "\t City %d, due to %d other cities, costing %0.3f" % (city, num_against, cost)
        # print "END --facility_disallowed_cities"
        # print "#############################"


# Taken from http://code.activestate.com/recipes/522995-priority-dict-a-priority-queue-with-updatable-prio/
import heapq
class priority_dict(dict):
    """Dictionary that can be used as a priority queue.

    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'

    The 'smallest' method can be used to return the object with lowest
    priority, and 'pop_smallest' also removes it.

    The 'sorted_iter' method provides a destructive sorted iterator.
    """

    def __init__(self, *args, **kwargs):
        super(priority_dict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.iteritems()]
        heapq.heapify(self._heap)

    def smallest(self):
        """Return the item with the lowest priority.

        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heapq.heappop(heap)
            v, k = heap[0]
        return k

    def pop_smallest(self):
        """Return the item with the lowest priority and remove it.

        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heapq.heappop(heap)
        while k not in self or self[k] != v:
            v, k = heapq.heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).

        super(priority_dict, self).__setitem__(key, val)

        if len(self._heap) < 2 * len(self):
            heapq.heappush(self._heap, (val, key))
        else:
            # When the heap grows larger than 2 * len(self), we rebuild it
            # from scratch to avoid wasting too much memory.
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.

        super(priority_dict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.

        Beware: this will destroy elements as they are returned.
        """

        while self:
            yield self.pop_smallest()



from numpy.random import np
from numpy import concatenate
import matplotlib.pyplot as plt
import math
def test_facility_location(numPts=20, ptDim=2, openCosts=None, p1=None, p2=None):
    """ Visualize the facility location solution.
    """
    # We'll run the experiment with various opening costs
    if openCosts is None:
        openCosts = [1,2,4,8,16,32,64,128,256]
        for i in range(0,len(openCosts)):
            openCosts[i] *= numPts/20.0

    # Generate some data points biased towards the top right
    if p1 is None:
        p1 = 2.5 * np.random.randn(numPts, ptDim) + 3

    # Generate some data points biased towards the bottom left
    if p2 is None:
        p2 = 2.5 * np.random.randn(numPts, ptDim) + -3
    p = np.concatenate((p1,p2))


    fac = FacilityLocation(points=p)



    w = math.ceil(math.sqrt(len(openCosts)))
    fig = plt.figure()

    for n in range(0,len(openCosts)):
        [facilities,cost]=fac.solve(openFacilityCost=openCosts[n])
        ax = fig.add_subplot(100*w + 10*w + n+1)
        pt1 = ax.plot(p1[:,0], p1[:,1], 'b.')
        pt2 = ax.plot(p2[:,0], p2[:,1], 'r.')
        for i in facilities:
            for j in facilities[i]:
                ax.plot([p[i,0],p[j,0]], [p[i,1],p[j,1]], 'g')
                ax.axes.get_xaxis().set_ticks([])
                ax.axes.get_yaxis().set_ticks([])
        ax.set_title('openCost=%0.3f totalCost=%0.3f' % (openCosts[n], cost) )
    fig.show()
    return (p1,p2)
