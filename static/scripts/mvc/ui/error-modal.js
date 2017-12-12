define("mvc/ui/error-modal", ["exports", "utils/localization"], function(exports, _localization) {
    "use strict";

    Object.defineProperty(exports, "__esModule", {
        value: true
    });

    var _localization2 = _interopRequireDefault(_localization);

    function _interopRequireDefault(obj) {
        return obj && obj.__esModule ? obj : {
            default: obj
        };
    }

    //TODO: toastr is another possibility - I didn't see where I might add details, tho

    /* ============================================================================
    Error modals meant to replace the o-so-easy alerts.
    
    These are currently styled as errormessages but use the Galaxy.modal
    infrastructure to be shown/closed. They're capable of showing details in a
    togglable dropdown and the details are formatted in a pre.
    
    Example:
        errorModal( 'Heres a message', 'A Title', { some_details: 'here' });
        errorModal( 'Heres a message' ); // no details, title is 'Error'
    
    There are three specialized forms:
        offlineErrorModal       a canned response for when there's no connection
        badGatewayErrorModal    canned response for when Galaxy is restarting
        ajaxErrorModal          plugable into any Backbone class as an
            error event handler by accepting the error args: model, xhr, options
    
    Examples:
        if( navigator.offLine ){ offlineErrorModal(); }
        if( xhr.status === 502 ){ badGatewayErrorModal(); }
        this.listenTo( this.model, 'error', ajaxErrorModal );
    
    ============================================================================ */

    var CONTACT_MSG = (0, _localization2.default)("Please contact a Galaxy administrator if the problem persists.");
    var DEFAULT_AJAX_ERR_MSG = (0, _localization2.default)("An error occurred while updating information with the server.");
    var DETAILS_MSG = (0, _localization2.default)("The following information can assist the developers in finding the source of the error:");

    /** private helper that builds the modal and handles adding details */
    function _errorModal(message, title, details) {
        // create and return the modal, adding details button only if needed
        Galaxy.modal.show({
            title: title,
            body: message,
            closing_events: true,
            buttons: {
                Ok: function Ok() {
                    Galaxy.modal.hide();
                }
            }
        });
        Galaxy.modal.$el.addClass("error-modal");

        if (details) {
            Galaxy.modal.$(".error-details").add(Galaxy.modal.$('button:contains("Details")')).remove();
            $("<div/>").addClass("error-details").hide().appendTo(Galaxy.modal.$(".modal-content")).append([$("<p/>").text(DETAILS_MSG), $("<pre/>").text(JSON.stringify(details, null, "  "))]);

            $("<button id=\"button-1\" class=\"pull-left\">" + (0, _localization2.default)("Details") + "</button>").appendTo(Galaxy.modal.$(".buttons")).click(function() {
                Galaxy.modal.$(".error-details").toggle();
            });
        }
        return Galaxy.modal;
    }

    /** Display a modal showing an error message but fallback to alert if there's no modal */
    function errorModal(message, title, details) {
        if (!message) {
            return;
        }

        message = (0, _localization2.default)(message);
        title = (0, _localization2.default)(title) || (0, _localization2.default)("Error:");
        if (window.Galaxy && Galaxy.modal) {
            return _errorModal(message, title, details);
        }

        alert(title + "\n\n" + message);
        console.log("error details:", JSON.stringify(details));
    }

    // ----------------------------------------------------------------------------
    /** display a modal when the user may be offline */
    function offlineErrorModal() {
        return errorModal((0, _localization2.default)("You appear to be offline. Please check your connection and try again."), (0, _localization2.default)("Offline?"));
    }

    // ----------------------------------------------------------------------------
    /** 502 messages that should be displayed when galaxy is restarting */
    function badGatewayErrorModal() {
        return errorModal((0, _localization2.default)("Galaxy is currently unreachable. Please try again in a few minutes.") + " " + CONTACT_MSG, (0, _localization2.default)("Cannot connect to Galaxy"));
    }

    // ----------------------------------------------------------------------------
    /** display a modal (with details) about a failed Backbone ajax operation */
    function ajaxErrorModal(model, xhr, options, message, title) {
        message = message || DEFAULT_AJAX_ERR_MSG;
        message += " " + CONTACT_MSG;
        title = title || (0, _localization2.default)("An error occurred");
        var details = _ajaxDetails(model, xhr, options);
        return errorModal(message, title, details);
    }

    /** build details which may help debugging the ajax call */
    function _ajaxDetails(model, xhr, options) {
        return {
            //TODO: still can't manage Raven id
            raven: _.result(window.Raven, "lastEventId"),
            userAgent: navigator.userAgent,
            onLine: navigator.onLine,
            version: _.result(Galaxy.config, "version_major"),
            xhr: _.omit(xhr, _.functions(xhr)),
            options: _.omit(options, "xhr"),
            // add ajax data from Galaxy object cache
            url: _.result(Galaxy.lastAjax, "url"),
            data: _.result(Galaxy.lastAjax, "data"),
            // backbone stuff (auto-redacting email for user)
            model: _.result(model, "toJSON", "" + model),
            user: _.omit(_.result(Galaxy.user, "toJSON"), "email")
        };
    }

    //=============================================================================
    exports.default = {
        errorModal: errorModal,
        offlineErrorModal: offlineErrorModal,
        badGatewayErrorModal: badGatewayErrorModal,
        ajaxErrorModal: ajaxErrorModal
    };
});
//# sourceMappingURL=../../../maps/mvc/ui/error-modal.js.map
