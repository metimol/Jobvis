/* ============================================================================
 * page-contact.js — behaviour for contact.html only.
 * vayra-shell.js is shared by every page; page specific code belongs here.
 *
 * This form has no backend, and that is deliberate. The template ships as static files.
 *   - Validation really runs: required fields, email shape, per field messages, aria-invalid.
 *   - Sending is not wired. The success message does not claim the message was sent,
 *     because until you add an endpoint it goes nowhere and the screen should say so.
 *   - The place to add your endpoint is marked in a comment above the form in contact.html.
 * ============================================================================ */

(function () {
  'use strict';

  /* Same test as the short form on index. Two forms in one product should not accept or
     reject the same address on different rules. */
  var EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

  var FIELDS = [
    {
      id: 'vy-entry-name',
      error: 'vy-entry-name-warn',
      test: function (v) { return v.length > 0; },
      message: 'Add a name so we know who is writing.'
    },
    {
      id: 'vy-entry-email',
      error: 'vy-entry-email-warn',
      test: function (v) { return EMAIL_RE.test(v); },
      message: 'That address does not look right.'
    },
    {
      id: 'vy-entry-message',
      error: 'vy-entry-message-warn',
      test: function (v) { return v.length > 0; },
      message: 'Tell us what you need. One line is enough.'
    }
  ];

  function boot() {
    var form = document.getElementById('vy-contact-form');
    var status = document.getElementById('vy-contact-status');
    if (!form || !status) return;

    /* Field handles are resolved once. A missing field simply drops out of the list, so
       deleting a field from the HTML does not stop the rest of the validation. */
    var checks = [];
    FIELDS.forEach(function (f) {
      var input = document.getElementById(f.id);
      var slot = document.getElementById(f.error);
      if (input && slot) checks.push({ input: input, slot: slot, test: f.test, message: f.message });
    });
    if (!checks.length) return;

    function clearField(c) {
      c.input.removeAttribute('aria-invalid');
      c.input.removeAttribute('aria-describedby');
      c.slot.textContent = '';
      c.slot.classList.remove('is-shown');
    }

    function failField(c) {
      c.input.setAttribute('aria-invalid', 'true');
      c.input.setAttribute('aria-describedby', c.slot.id);
      c.slot.textContent = c.message;
      c.slot.classList.add('is-shown');
    }

    function setStatus(text) {
      status.textContent = text;
      status.classList.toggle('is-shown', text.length > 0);
    }

    checks.forEach(function (c) {
      /* Editing an input clears that field's error and leaves the others. Clearing every
         message on the first keystroke hides what is still wrong. */
      c.input.addEventListener('input', function () {
        if (c.input.hasAttribute('aria-invalid')) clearField(c);
        if (status.textContent) setStatus('');
      });
    });

    form.addEventListener('submit', function (e) {
      e.preventDefault();

      var first = null;
      checks.forEach(function (c) {
        if (c.test(c.input.value.trim())) {
          clearField(c);
        } else {
          failField(c);
          if (!first) first = c.input;
        }
      });

      if (first) {
        setStatus('Check the fields marked above.');
        first.focus();
        return;
      }

      /* This is the send point. It does not send, and it does not pretend to.
         Once you put an action on the form, either remove this preventDefault or put
         fetch(form.action, { method: 'post', body: new FormData(form) }) here. */
      setStatus('Validated. Connect this form to your own endpoint before launch.');
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
