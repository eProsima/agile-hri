# This is based on the intent msg done in https://github.com/ros4hri/hri_actions_msgs/tree/humble-devel

# An intent is an abstract description of an operation to be performed by the
# robot.
#
# While inspired by the Android intents
# (https://developer.android.com/reference/android/content/Intent), ROS intents
# are primarily designed to capture user-initiated intents. For instance, a
# button click on a touchscreen, the result of a chatbot-based verbal
# interaction, a command started by a remote user interface.
#
# Intents comprise of two mandatory fields: the `intent`, which should be one
# of the available predefined intents, and the `data` which must be a JSON
# object containing the data required to fully instantiate the intent.  While
# the exact content of the JSON object is up to the application, the following
# standard keys can be used to specify the intent's semantic roles (also called
# thematic roles or theta roles):
#
# - `agent`: the agent expected to perform the intent (if omitted, the robot
# itself is assumed)
# - `object` (also named *theme* or *patient* in the linguistics literature): entity
# undergoing the effect of the intent
# - `goal`: entity towards which the intent is directed or moves
# - `recipient`: entity that receives the object (denoted as the `theme`)
#
# Examples:
#
# - "I want you to go to the kitchen":
#   - intent: `move_to`
#   - data: {"goal":"kitchen_1"}
#
# - "Can you take the groceries to Luke in the kitchen?"
#   - intent: `bring`
#   - data: {"object": "groceries",
#            "goal":"kitchen_1",
#            "recipient": "person_luke"}

##########################################################################
#  Intent name
#
# A string describing the intent.
#
# Where suitable, the intent SHOULD be one of the constant defined below.
# However, we recognise that the list of intents is possibly large. Therefore,
# custom strings are also permissible. If you believe your intent name should
# be standardised and added to the list of pre-defined intents, fill the
# corresponding entry in the "thematic roles" table below and submit a pull
# request on this repository.
string intent

# 'intent' constants
# see table below for a short description of each.

string BRING_OBJECT = __intent_bring_object__
string DANCE = __intent_dance__
string DESCRIBE_ENV = __intent_describe_env__
string ENGAGE_WITH = __intent_engage_with__
string GRAB_OBJECT = __intent_grab_object__
string GREET = __intent_greet__
string GUIDE = __intent_guide__
string INTRODUCTION = __intent_introduce_yourself__
string MOVE_TO = __intent_move_to__
string OTHER = __intent_other_action__
string PERFORM_MOTION = __intent_perform_motion__
string PLACE_OBJECT = __intent_place_object__
string PRESENT_CONTENT = __intent_present_content__
string SAY = __intent_say__
string START_ACTIVITY = __intent_start_activity__
string STOP_ACTIVITY = __intent_stop_activity__
string SUSPEND = __intent_suspend__
string WAKEUP = __intent_wakeup__


##########################################################################
#  Intent data
#
# a JSON object containing the data required to fully instantiate the intent.
# The keys of the object should be one of the thematic role defined above, or
# the general `other_data`.
#
# Each intent defines a specific set of required and optional thematic roles,
# listed in the following table (note that the `agent` role can be optionally
# added to all intents, and is omitted from the table for clarity):
#
# .. [please keep the RST table syntax below, as it is used to automatically parse the list of intents by some nodes]#
# .. BEGIN_INTENTS_TABLE
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | **Intent**       | **Description**                                                      | **Required thematic roles**                  | **Optional thematic roles**                                               |
# +==================+======================================================================+==============================================+===========================================================================+
# | `BRING_OBJECT`   | bring a specific object to a specific place                          | - `object`                                   |                                                                           |
# |                  |                                                                      | - `recipient`                                |                                                                           |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | `DANCE`          | start a dance action                                                 |                                              |                                                                           |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | `DESCRIBE_ENV`   | analyze and describe the robot environment                           |                                              |                                                                           |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | `ENGAGE_WITH`    | an agent wants to engage with another one                            | - `recipient`                                |                                                                           |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | `GRAB_OBJECT`    | pick-up a specific object                                            | - `object`                                   |                                                                           |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | `GREET`          | greet an agent                                                       | - `recipient`                                |                                                                           |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | `GUIDE`          | guides someone somewhere                                             | - `goal`                                     |                                                                           |
# |                  |                                                                      | - `recipient`                                |                                                                           |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | `INTRODUCTION`   | introduce your self                                                  |                                              |                                                                           |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | `MOVE_TO`        | navigates to a specific location                                     | - `goal`                                     |                                                                           |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | `OTHER`          | unknown intent                                                       |                                              |                                                                           |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | `PERFORM_MOTION` | performs a motion (e.g., a specific gesture like pointing, waving)   | - `object` (the name of the motion/gesture)  | - `recipient`                                                             |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | `PLACE_OBJECT`   | put an object on a support (e.g. a table)                            | - `recipient`                                | - `object` (if more that one object could be placed)                      |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | `PRESENT_CONTENT`| present (via a screen, pre-recorded text...) predefined content      | - `object` (the content identifier)          | - `recipient`                                                             |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | `SAY`            | says some text, optionally annotated with gestures or expressions    | - `object` (the text to say)                 | - `recipient`                                                             |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | `START_ACTIVITY` | start a scripted behaviour/activity                                  | - `object` (the name of the activity)        | - any condition to start the activity                                     |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | `STOP_ACTIVITY`  | request cancelation of an activity                                   |                                              | - `object` (the name of the activity. If unset, current main activity)    |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | `SUSPEND`        | request suspension of the system                                     |                                              |                                                                           |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# | `WAKEUP`         | request reactivation of the system                                   |                                              |                                                                           |
# +------------------+----------------------------------------------------------------------+----------------------------------------------+---------------------------------------------------------------------------+
# .. END_INTENTS_TABLE
#
string data

##########################################################################
#  Intent source
#
# A string describing the source of the intent. This is *not* the node
# which published the intent, but instead the actual agent who expressed the
# intent/command/desire.
# `source` can be either one of the constant below, or the specific id of the
# person/agent expressing the intent. In a REP-155 compliant system, this ID
# must be the person ID of the agent.
string source

# 'source' constants:

# For intents originating from the robot itself
string ROBOT_ITSELF = __myself__
# For intents originating from a external robot control system (for instance, a remote control tablet)
string REMOTE_SUPERVISOR = __remote_supervisor__
# For intents coming from an agent interacting with the robot, but not uniquely
# identified
string UNKNOWN_AGENT = __unknown_agent__
# For unknown sources
string UNKNOWN = __unknown__


##########################################################################
#  Intent modality
#
# Modality used to originally convey the intent: verbal, via the touchscreen,
# via a gesture...
# The special modality MODALITY_INTERNAL must be used for intents coming for the
# robot's internal processes (e.g. 'battery low, I need to charge')
# MUST be one of the MODALITY_ constant below.
string modality

# 'modality' constants:

string MODALITY_SPEECH = __modality_speech__
# both gestures (e.g. waving) and displacement (e.g. approaching)
string MODALITY_MOTION = __modality_motion__
string MODALITY_TOUCHSCREEN = __modality_touchscreen__
string MODALITY_OTHER = __modality_other__
string MODALITY_INTERNAL = __modality_internal__


##########################################################################
#  Intent priority
#
# The priority of this intent. This MIGHT be used as a hint by the robot's task
# scheduler, however is scheduler is *not* forced to respect this priority
# level. 0 is the lowest priority, 128 is the default priority, 255 is the
# highest priority.
uint8 priority

##########################################################################
#  Intent confidence
#
# A value between 0.0 (no confidence) and 1.0 (full confidence) that the intent
# was correctly perceived and interpreted.
#
# For instance, a 'waving' gesture could be interpreted as an implicit request
# from a user for the robot to greet back or engage. As this interpretation is
# not certain, the confidence of the intent may be below 1.0.
float32 confidence
