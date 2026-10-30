"""Tracking identity, complete-view sampling and temporal refusal regressions."""
import unittest
import numpy as np
from src.tracking import FruitTracker,TemporalVote,Track,interior_observation


def prediction(fruit='banana',stage='ripe'):
    return {'status':'ok','fruit':fruit,'ripeness':stage,'probabilities':{stage:.9},
        'agent_decision':{'action':'auto_classify'}}


def unknown():
    return {'status':'unknown','fruit':'unknown','ripeness':None,'candidate_fruit':'mandarin','candidate_ripeness':'ripe'}


def track():
    return Track(1,(100,100,100,100),0,np.ones(144)/12)


class TemporalTests(unittest.TestCase):
    def test_minimum_evidence_and_one_wrong_frame_do_not_flip_label(self):
        t=track();vote=TemporalVote(30)
        for frame in range(4):
            vote.add(t,prediction(),frame)
        self.assertEqual(vote.display(t,3,True)['status'],'pending')
        vote.add(t,prediction(),4)
        self.assertEqual(vote.display(t,4,True)['fruit'],'banana')
        vote.add(t,prediction('tomato'),5)
        self.assertEqual(vote.display(t,5,True)['fruit'],'banana')
        self.assertEqual(t.switches,0)

    def test_consistent_unknown_is_never_promoted_to_known_candidate(self):
        t=track();vote=TemporalVote(30)
        for frame in range(10):
            vote.add(t,unknown(),frame)
        result=vote.display(t,9,True)
        self.assertEqual(result['status'],'unknown')
        self.assertIsNone(result['ripeness'])

    def test_sustained_change_eventually_replaces_old_label(self):
        t=track();vote=TemporalVote(30,min_samples=3,window=5)
        for frame in range(5):
            vote.add(t,prediction(),frame)
        for frame in range(5,13):
            vote.add(t,unknown(),frame)
        self.assertEqual(vote.display(t,12,True)['status'],'unknown')
        self.assertEqual(t.switches,1)

    def test_old_evidence_expires_and_edge_never_requests_samples(self):
        t=track();vote=TemporalVote(30)
        for frame in range(5):
            vote.add(t,prediction(),frame)
        self.assertFalse(vote.needs_sample(t,20,False))
        self.assertEqual(vote.display(t,50,False)['status'],'pending')

    def test_only_interior_size_stable_observations_are_eligible(self):
        t=track()
        t.box=(0,100,100,100)
        self.assertFalse(interior_observation(t,(400,600,3)))
        t.box=(100,100,100,100)
        self.assertFalse(interior_observation(t,(400,600,3)))
        self.assertFalse(interior_observation(t,(400,600,3)))
        self.assertTrue(interior_observation(t,(400,600,3)))
        t.box=(100,100,50,50)
        self.assertFalse(interior_observation(t,(400,600,3)))


class TrackerTests(unittest.TestCase):
    def image(self,boxes,colors):
        frame=np.full((300,700,3),255,np.uint8)
        for (x,y,w,h),color in zip(boxes,colors):
            frame[y:y+h,x:x+w]=color
        return frame

    def test_two_moving_objects_keep_separate_ids(self):
        tracker=FruitTracker()
        boxes=[(50,80,60,60),(300,80,60,60)]
        colors=[(0,0,200),(0,180,0)]
        first=tracker.update(boxes,self.image(boxes,colors),0)
        moved=[(58,80,60,60),(308,80,60,60)]
        second=tracker.update(moved,self.image(moved,colors),1)
        self.assertEqual([t.id for t in first],[t.id for t in second])
        self.assertNotEqual(second[0].id,second[1].id)

    def test_disappeared_objects_not_drawn_and_expired_id_not_reused(self):
        tracker=FruitTracker(max_missed=2)
        boxes=[(50,80,60,60)];colors=[(0,0,200)]
        first=tracker.update(boxes,self.image(boxes,colors),0)[0]
        self.assertEqual(tracker.update([],self.image([],[]),1),[])
        new=tracker.update(boxes,self.image(boxes,colors),4)[0]
        self.assertNotEqual(first.id,new.id)
        self.assertIsNone(new.stable)

    def test_replacement_with_different_appearance_does_not_inherit_vote(self):
        tracker=FruitTracker();boxes=[(50,80,60,60)]
        first=tracker.update(boxes,self.image(boxes,[(0,0,200)]),0)[0]
        first.stable=('ok','tomato','ripe')
        second=tracker.update(boxes,self.image(boxes,[(0,180,0)]),1)[0]
        self.assertNotEqual(first.id,second.id)
        self.assertIsNone(second.stable)


if __name__=='__main__':
    unittest.main()
