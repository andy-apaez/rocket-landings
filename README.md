# Rocket Landings

A rocket simulator where a computer teaches itself to land a rocket.

You launch a rocket from Earth, put it into orbit and pick anywhere on the globe to land. Then you watch an AI
pilot try to bring it down safely. At first it crashes. Every time you launch, it practices thousands of times
behind the scenes and gets a little better, until it can set the rocket down gently on its landing legs.

## What you can do

**Fly a full mission.** Pick a launch site and how high you want to go, then press _Lift off_. You'll see a
cinematic countdown and launch from the pad, then follow the rocket into orbit on a live 3D globe. Once you're in
orbit, click any spot on land to choose where to come down. The rocket fires its engines to drop out of orbit,
survives the fiery trip back through the atmosphere, and hands over to the AI pilot for the final landing.

**Watch the AI learn.** In _Classic launch control_ you set up the situation yourself: how high the rocket starts,
how fast it's falling, how far it is from the landing pad, how much fuel is left, and whether it's on Earth, Mars
or the Moon. Each launch trains the pilot a bit more, and a chart shows it improving over time. You can also
switch to the built-in _Autopilot_, a hand-written controller that needs no training, to compare.

**Try a tower catch.** Instead of landing on legs, the rocket has to hover between two giant arms on a tower and
hold still while they close around it. It's much harder.

## How it works, in plain terms

- **The physics is realistic.** Gravity, air resistance, wind, fuel getting used up, and an engine that can tilt
  to steer are all simulated. Mars and the Moon have their own gravity and thinner (or no) air, so each one flies
  differently.
- **The pilot learns by trial and error.** It is never told how to land. It only gets rewarded for getting close
  to the pad, staying upright and touching down softly, and penalized for crashing. Over millions of practice
  attempts it figures out the rest.
- **The landing site matters.** The ground is generated from the spot you picked on the globe, so mountains are
  rough, flat areas are smooth, and the wind depends on where in the world you are.

## The Blueprint before the build.

My personal blueprint for this project can be found in notes.md. This project is heavily inspired by spacex and its Starship, but any type of rocket model and its specs can easily be implimented into this along with varius planets and how they affect a landing.

A few AI agents were used in the process such as Claude Opus 5.5 (medium effort), Gemini 4 Argon (medium thinking), and Claude Haiku 5.5 (medium effort). Each had their own tasks, but they were mainily used in the physics process, espcially when gathering specs for planets and rocket models. Gemini 4 Argon had its role primarily focused on the rocket thrusters and their phyics while Claude Opus 5.5 worked on models and planet physics. Haiku was used for the mathematics section.

An opensource neuro network is being used for the training, but if you'd like a from scratch network can be implimented if you take the time to make it.

## Controls

| Key           | What it does                   |
| ------------- | ------------------------------ |
| Space         | Pause                          |
| L             | Launch (classic mode)          |
| ← / →         | Switch between rockets         |
| P             | View the landing pad and tower |
| R             | Zoom in on the selected rocket |
| Drag / scroll | Spin and zoom the globe        |

## Getting started

You'll need [Python](https://www.python.org/downloads/) 3.10 or newer and a web browser.

```bash
# copy the project to your computer
git clone https://github.com/andy-apaez/rocket-landings.git
cd rocket-landings

# install what it needs
pip install -r requirements.txt

# start it (opens http://localhost:8000/viewer/ in your browser)
python3 play.py
```

To start over with a brand-new, untrained pilot:

```bash
python3 play.py --fresh
```

For developers:

```bash
python3 train.py train --steps 5000000   # train the pilot for a long session
pytest3 tests                            # run the tests
```
