/**
 * Simple stakeholder form (DMCH + RMCH) for the hospital robot's cultural rules.
 *
 * Deploy: script.google.com -> New project -> paste this -> Run buildForm()
 * -> authorize -> the form link is printed in the log and appears in Drive.
 */

const RATING = ['About right', 'Too strict', 'Should be stronger'];

const RULES = [
  'During prayer times, the robot never passes through the prayer room, even if it is the shortest way.',
  'Outside prayer times, the robot may pass through the empty prayer room.',
  'The robot enters a gender-segregated ward only for a task, goes directly, and leaves. It never waits inside.',
  'The robot keeps extra distance from family members around a patient’s bed and never drives between them and the patient.',
  'The robot slows down and keeps clear of beds placed in corridors.',
];

function buildForm() {
  const form = FormApp.create('Hospital Delivery Robot — Are These Rules Right?');

  form.setDescription(
    'A small robot is being developed to carry medicine and supplies through ' +
    'hospital corridors. It follows fixed rules for sensitive places. Please ' +
    'tell us if each rule is right for your hospital. Takes 5 minutes. Anonymous.');
  form.setCollectEmail(false);

  form.addMultipleChoiceItem()
    .setTitle('Your hospital')
    .setChoiceValues(['DMCH (Dhaka)', 'RMCH (Rajshahi)'])
    .setRequired(true);

  form.addMultipleChoiceItem()
    .setTitle('You are')
    .setChoiceValues([
      'Doctor / nurse / ward staff',
      'Administration',
      'Imam / prayer-space caretaker',
      'Patient’s family member',
    ])
    .showOtherOption(true)
    .setRequired(true);

  RULES.forEach(function (text, i) {
    form.addMultipleChoiceItem()
      .setTitle('Rule ' + (i + 1) + ': ' + text)
      .setChoiceValues(RATING)
      .setRequired(true);
  });

  form.addMultipleChoiceItem()
    .setTitle('For an urgent medicine delivery, how much extra time is acceptable to follow these rules?')
    .setChoiceValues(['None', 'Up to 30 seconds', 'Up to 2 minutes', 'Any — never break the rules'])
    .setRequired(true);

  form.addParagraphTextItem()
    .setTitle('Anything we missed, or any rule you would change? (optional)');

  form.addScaleItem()
    .setTitle('How comfortable would you be with this robot in your area?')
    .setBounds(1, 5)
    .setLabels('Not at all', 'Fully')
    .setRequired(true);

  Logger.log('Edit link:  ' + form.getEditUrl());
  Logger.log('Share link: ' + form.getPublishedUrl());
}
