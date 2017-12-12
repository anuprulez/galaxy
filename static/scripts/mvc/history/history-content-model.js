define("mvc/history/history-content-model", ["exports", "mvc/dataset/states", "mvc/base-mvc", "utils/localization"], function(exports, _states, _baseMvc, _localization) {
    "use strict";

    Object.defineProperty(exports, "__esModule", {
        value: true
    });

    var _states2 = _interopRequireDefault(_states);

    var _baseMvc2 = _interopRequireDefault(_baseMvc);

    var _localization2 = _interopRequireDefault(_localization);

    function _interopRequireDefault(obj) {
        return obj && obj.__esModule ? obj : {
            default: obj
        };
    }

    var collectionFuzzyCountDefault = 1000;
    try {
        collectionFuzzyCountDefault = localStorage.getItem("collectionFuzzyCountDefault") || collectionFuzzyCountDefault;
    } catch (err) {}

    //==============================================================================
    /** @class Mixin for HistoryContents content (HDAs, HDCAs).
     */
    var HistoryContentMixin = {
        /** default attributes for a model */
        defaults: {
            /** parent (containing) history */
            history_id: null,
            /** some content_type (HistoryContents can contain mixed model classes) */
            history_content_type: null,
            /** indicating when/what order the content was generated in the context of the history */
            hid: null,
            /** whether the user wants the content shown (visible) */
            visible: true
        },

        // ........................................................................ mixed content element
        // In order to be part of a MIXED bbone collection, we can't rely on the id
        //  (which may collide btwn models of different classes)
        // Instead, use type_id which prefixes the history_content_type so the bbone collection can differentiate
        idAttribute: "type_id",

        // ........................................................................ common queries
        /** the more common alias of visible */
        hidden: function hidden() {
            return !this.get("visible");
        },

        //TODO: remove
        /** based on includeDeleted, includeHidden (gen. from the container control),
         *      would this ds show in the list of ds's?
         *  @param {Boolean} includeDeleted are we showing deleted hdas?
         *  @param {Boolean} includeHidden are we showing hidden hdas?
         */
        isVisible: function isVisible(includeDeleted, includeHidden) {
            var isVisible = true;
            if (!includeDeleted && (this.get("deleted") || this.get("purged"))) {
                isVisible = false;
            }
            if (!includeHidden && !this.get("visible")) {
                isVisible = false;
            }
            return isVisible;
        },

        // ........................................................................ ajax
        //TODO?: these are probably better done on the leaf classes
        /** history content goes through the 'api/histories' API */
        urlRoot: Galaxy.root + "api/histories/",

        /** full url spec. for this content */
        url: function url() {
            var historyContentType = this.get("history_content_type");
            var historyId = this.get("history_id");
            var historyContentId = this.get("id");
            var url = "" + this.urlRoot + historyId + "/contents/" + historyContentType + "s/" + historyContentId;
            if (historyContentType == "dataset_collection") {
                // Don't fetch whole collection - just enought to render outline. Backbone will
                // make a detailed request if any datasets are expanded beyond that point.
                url = url + "?view=element-reference&fuzzy_count=" + collectionFuzzyCountDefault;
            }
            return url;
        },

        /** save this content as not visible */
        hide: function hide(options) {
            if (!this.get("visible")) {
                return jQuery.when();
            }
            return this.save({
                visible: false
            }, options);
        },
        /** save this content as visible */
        unhide: function unhide(options) {
            if (this.get("visible")) {
                return jQuery.when();
            }
            return this.save({
                visible: true
            }, options);
        },

        // ........................................................................ misc
        toString: function toString() {
            return [this.get("type_id"), this.get("hid"), this.get("name")].join(":");
        }
    };

    //==============================================================================
    exports.default = {
        HistoryContentMixin: HistoryContentMixin
    };
});
//# sourceMappingURL=../../../maps/mvc/history/history-content-model.js.map
