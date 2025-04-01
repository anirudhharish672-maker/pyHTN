from abc import ABCMeta


class BaseCondLearner(metaclass=ABCMeta):
    def __init__(self, skill, **kwargs):
        """

        :param skill:
        :param kwargs:
        """

        self.skill = skill
        self.check_sanity = kwargs.get('sanity_check', True)

        # Note this line makes it possible to call
        # super(BaseCondLearner, self).__init__(skill, **kwargs)

    def sanity_check_ifit(self, state, skill_app, reward):
        # SANITY CHECK: Classifier Inconsistencies
        if self.check_sanity and reward is not None:
            prediction = self.predict(state, skill_app.match)
            if reward != prediction:
                raise Exception(f"(Sanity Check Error): Condition-learning mechanism" +
                                f" for skill {self.skill}"
                                f" .predict() produces different outcome ({prediction:.2f}) than reward given in" +
                                f" .ifit() ({reward:.2f}). This most likely indicates 1) that the agent" +
                                " requires additional feature prior knowledge to distinguish between two" +
                                " now indistinguishable situations. Alternatively 2) this may indicate an error in the " +
                                " when-learning mechanism or in the preparation of the state representation.")

    def __init_subclass__(cls, **kwargs):
        """

        :param kwargs:
        :return:
        """
        super().__init_subclass__(**kwargs)

        # Inject sanity checks after ifit
        ifit = cls.ifit

        def ifit_w_sanity_check(self, state, skill_app, reward):
            ifit(self, state, skill_app, reward)
            if self.check_sanity:
                self.sanity_check_ifit(state, skill_app, reward)

        setattr(cls, 'ifit', ifit_w_sanity_check)

    def ifit(self, state, skill_app, reward):
        """

        :param state:
        :param skill_app:
        :param reward:
        :return:
        """
        raise NotImplemented()

    def fit(self, states, skill_apps, reward):
        """

        :param states:
        :param skill_apps:
        :param reward:
        :return:
        """

        raise NotImplemented()

    def score(self, state, skill_app):
        """

        :param state:
        :param skill_app:
        :return:
        """
        raise NotImplemented()

    def as_conditions(self):
        """

        :return:
        """
        raise NotImplemented()

    def predict(self, state: list[dict], match: tuple):
        """

        :param state:
        :param match:
        :return:
        """
        raise NotImplemented()

    def get_info(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        return {}

    def get_pyhtn_conds(self) -> list[tuple]:
        """
        Provides a representation of conditions in DNF (A disjunction of conjunctions of literals)
        formatted as [...[...[...literals]]].
        :return: List of tuples
        """
        raise NotImplemented()

    def get_lit_priorities(self) -> list[tuple[float, tuple]]:
        """
        Outputs a list of tuples like [...(priority, literal)] where priority is in [0.0,1.0] representing the
        priority value of a condition element.
        :return:
        """
        raise NotImplemented()

