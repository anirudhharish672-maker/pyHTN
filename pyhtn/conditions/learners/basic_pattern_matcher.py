from pyhtn.conditions.learners.condition_learner_interface import BaseCondLearner


class BasicPatternMatcher(BaseCondLearner):
    def __init__(self, skill, **kwargs):
        super().__init__(skill, **kwargs)

